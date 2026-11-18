"""Phase 1 data collection.

Runs a proposer through ALFWorld training games. At every step it records the
outcome and label of every candidate (by replay), then executes the proposer's
first choice, or with probability --explore a random candidate.

    uv run python -m src.collect --run NAME [--model claude-haiku-5-5] [--games-per-type 25]
    uv run python -m src.collect --run NAME --provider openai --model MODEL [--base-url URL]
    uv run python -m src.collect --run smoke --stub --games-per-type 2     # pipeline test, no LLM
    uv run python -m src.collect --run NAME --summary                      # stats for an existing run

Output: data/logs/NAME/steps.jsonl, one JSON object per step. Re-running the
same command resumes, skipping games already logged.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

from .agent import HISTORY_STEPS, AnthropicProposer, LLMProposer, StubProposer
from .env import ERROR, USEFUL, AlfEnv, Game, TASK_TYPES, list_games

LOG_ROOT = Path(__file__).resolve().parent.parent / "data" / "logs"


def run_episode(game: Game, proposer, explore: float, max_steps: int, seed: int) -> list[dict]:
    rng = random.Random(f"{seed}/{game.game_id}")
    env = AlfEnv(game)
    obs = env.reset()
    room = obs.feedback
    history: list[tuple[str, str]] = []
    rows = []
    for step in range(max_steps):
        if obs.won:
            break
        t = time.time()
        cands = proposer.propose(env.goal, history, obs.feedback, room=room,
                                 expert_action=obs.next_expert_action, admissible=obs.admissible)
        t_propose = time.time() - t
        outcomes = env.branch(cands)
        chosen = rng.randrange(len(cands)) if rng.random() < explore else 0
        rows.append({
            "game_id": game.game_id,
            "task_type": game.task_type,
            "step": step,
            "goal": env.goal,
            "room": room,
            "observation": obs.feedback,
            "history": history[-HISTORY_STEPS:],
            "steps_to_go": obs.steps_to_go,
            "expert_action": obs.next_expert_action,
            "candidates": [
                {"action": o.action, "rank": i, "outcome": o.feedback, "steps_after": o.steps_after,
                 "won": o.won, "label": o.label, "executed": i == chosen}
                for i, o in enumerate(outcomes)
            ],
            "explored": chosen != 0,
            "propose_s": round(t_propose, 3),
        })
        out = env.step(cands[chosen])
        history.append((out.action, out.feedback))
        obs = env.obs
    env.close()
    for r in rows:
        r["episode_won"] = obs.won
        r["episode_steps"] = len(rows)
    return rows


_worker_cfg: dict = {}


def _init_worker(cfg):
    _worker_cfg.update(cfg)
    if cfg["stub"]:
        _worker_cfg["proposer"] = StubProposer(n=cfg["n"], seed=cfg["seed"] + mp.current_process().pid)
    elif cfg["provider"] == "anthropic":
        _worker_cfg["proposer"] = AnthropicProposer(model=cfg["model"], n=cfg["n"], effort=cfg["effort"],
                                                    thinking=cfg["thinking"])
    else:
        _worker_cfg["proposer"] = LLMProposer(model=cfg["model"], n=cfg["n"], temperature=cfg["temperature"],
                                              base_url=cfg["base_url"], thinking=cfg["thinking"],
                                              min_interval=60 * cfg["workers"] / cfg["rpm"] if cfg["rpm"] else 0,
                                              api_key=os.environ.get(cfg["api_key_env"]) if cfg["api_key_env"] else None)


def _work(game: Game):
    c = _worker_cfg
    p = c["proposer"]
    before = (getattr(p, "calls", 0), getattr(p, "prompt_tokens", 0), getattr(p, "completion_tokens", 0))
    try:
        rows = run_episode(game, p, c["explore"], c["max_steps"], c["seed"])
    except Exception as e:  # one bad game or API failure should not end the run
        return game.game_id, [], f"{type(e).__name__}: {e}", (0, 0, 0)
    after = (getattr(p, "calls", 0), getattr(p, "prompt_tokens", 0), getattr(p, "completion_tokens", 0))
    return game.game_id, rows, None, tuple(a - b for a, b in zip(after, before))


def pick_games(per_type: int, seed: int) -> list[Game]:
    rng = random.Random(seed)
    games = list_games("train")
    return [g for tt in TASK_TYPES for g in rng.sample([g for g in games if g.task_type == tt], per_type)]


def load_steps(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def summarize(rows: list[dict]) -> str:
    if not rows:
        return "no steps logged"
    episodes = {r["game_id"]: r for r in rows}
    labels = Counter(c["label"] for r in rows for c in r["candidates"])
    n_cand = sum(labels.values())
    top1 = sum(r["candidates"][0]["label"] == USEFUL for r in rows)
    any_useful = sum(any(c["label"] == USEFUL for c in r["candidates"]) for r in rows)
    top1_err = sum(r["candidates"][0]["label"] == ERROR for r in rows)
    n = len(rows)
    lines = [
        f"episodes {len(episodes)}, steps {n}, labelled candidates {n_cand} "
        f"({n_cand / n:.1f} per step)",
        f"episode success {sum(e['episode_won'] for e in episodes.values()) / len(episodes):.0%}",
        "candidate labels: " + ", ".join(f"{k} {v / n_cand:.0%}" for k, v in labels.most_common()),
        f"first choice useful          {top1 / n:.1%}",
        f"first choice an error        {top1_err / n:.1%}",
        f"some candidate useful        {any_useful / n:.1%}   (oracle)",
        f"headroom (oracle - first)    {(any_useful - top1) / n:.1%}",
    ]
    by_type = defaultdict(list)
    for r in rows:
        by_type[r["task_type"]].append(r)
    lines.append(f"\n{'task type':32s} episodes  steps  won  first useful  oracle")
    for tt in TASK_TYPES:
        rs = by_type.get(tt, [])
        if not rs:
            continue
        eps = {r["game_id"]: r["episode_won"] for r in rs}
        lines.append(f"{tt:32s} {len(eps):8d} {len(rs):6d} {sum(eps.values()) / len(eps):4.0%} "
                     f"{sum(r['candidates'][0]['label'] == USEFUL for r in rs) / len(rs):12.0%} "
                     f"{sum(any(c['label'] == USEFUL for c in r['candidates']) for r in rs) / len(rs):7.0%}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="name of the run; logs go to data/logs/RUN/")
    ap.add_argument("--provider", choices=("anthropic", "openai"), default="anthropic",
                    help="anthropic: Claude API via the Anthropic SDK; openai: any OpenAI-compatible endpoint")
    ap.add_argument("--model", help="model id (default claude-haiku-5-5 for anthropic)")
    ap.add_argument("--base-url", help="openai provider only, e.g. http://localhost:11434/v1 for Ollama")
    ap.add_argument("--api-key-env", help="openai provider only: name of the env var holding the key for "
                    "--base-url, e.g. DEEPSEEK_API_KEY (not needed for Ollama)")
    ap.add_argument("--effort", default="low", help="anthropic provider only")
    ap.add_argument("--thinking", action="store_true", help="let the model reason before answering; off by default")
    ap.add_argument("--rpm", type=float, default=0, help="openai provider only: requests-per-minute cap across all workers")
    ap.add_argument("--stub", action="store_true", help="pipeline test with no LLM")
    ap.add_argument("--summary", action="store_true", help="print stats for the run and exit")
    ap.add_argument("--games-per-type", type=int, default=25)
    ap.add_argument("--n", type=int, default=5, help="candidates per step")
    ap.add_argument("--temperature", type=float, default=0.7, help="openai provider only")
    ap.add_argument("--explore", type=float, default=0.25, help="chance of executing a random candidate")
    ap.add_argument("--max-steps", type=int, default=40)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = LOG_ROOT / args.run
    steps_path = out_dir / "steps.jsonl"
    if args.summary:
        print(summarize(load_steps(steps_path)))
        return
    if not args.stub and not args.model:
        if args.provider != "anthropic":
            ap.error("give --model, or --stub for a pipeline test")
        args.model = "claude-haiku-5-5"

    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = {k: getattr(args, k) for k in
           ("provider", "model", "base_url", "api_key_env", "effort", "thinking", "rpm", "workers", "stub", "n", "temperature", "explore", "max_steps", "seed", "games_per_type")}
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    done = {r["game_id"] for r in load_steps(steps_path)}
    games = [g for g in pick_games(args.games_per_type, args.seed) if g.game_id not in done]
    print(f"{len(done)} games already logged, {len(games)} to run, {args.workers} workers", flush=True)

    t0, calls, ptok, ctok, n_steps, failed = time.time(), 0, 0, 0, 0, 0
    with mp.Pool(args.workers, initializer=_init_worker, initargs=(cfg,)) as pool, open(steps_path, "a") as f:
        for i, (gid, rows, err, usage) in enumerate(pool.imap_unordered(_work, games, chunksize=1), 1):
            if err:
                failed += 1
                print(f"FAILED {gid}: {err}", flush=True)
                continue
            for r in rows:
                f.write(json.dumps(r) + "\n")
            f.flush()
            n_steps += len(rows)
            calls, ptok, ctok = calls + usage[0], ptok + usage[1], ctok + usage[2]
            if i % 10 == 0 or i == len(games):
                print(f"{i}/{len(games)} games, {n_steps} steps, {time.time() - t0:.0f}s, "
                      f"{calls} LLM calls, {ptok} prompt + {ctok} completion tokens", flush=True)

    print(f"\ndone: {failed} games failed\n")
    print(summarize(load_steps(steps_path)))


if __name__ == "__main__":
    main()
