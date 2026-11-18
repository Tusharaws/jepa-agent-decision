"""Phase 4: play whole games with each selector choosing the action.

    uv run python -m src.eval_online --train-run qwen38-main --run online-main \
        --provider openai --model Qwen3.8-27B --base-url URL --api-key-env AMD_API_KEY --rpm 18 --workers 2
    uv run python -m src.eval_online --train-run NAME --run NAME2 --summary
    uv run python -m src.eval_online --train-run NAME --run smoke --stub --games 4     # no LLM

Arms (same games, same proposer, same prompt):

    llm_first       execute the proposer's first choice
    decision_rank   decision model picks among the candidates
    jepa_rank       predictor + value head picks
    hybrid_rank     world model filters, decision model picks

The selectors are trained once on every game of --train-run (85 / 15 train /
validation by game) and saved to results/RUN/models.pt.

Within a game the proposer's answer is cached by prompt, so two arms that are
in the same situation see the same candidates; they only differ from the step
where a selector picks something else. This pairs the arms and saves calls.

Output: data/logs/RUN/online.jsonl, one JSON object per (game, arm), and
results/RUN/online.json. Re-running resumes, skipping games already logged.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .agent import HISTORY_STEPS, AnthropicProposer, LLMProposer, StubProposer, build_user_prompt
from .collect import LOG_ROOT
from .env import ERROR, USEFUL, AlfEnv, Game, TASK_TYPES, list_games

RESULTS = Path(__file__).resolve().parent.parent / "results"
ARMS = ("llm_first", "decision_rank", "jepa_rank", "hybrid_rank")
SAVED = ("predictor", "value_pred_rank", "decision_rank")


def train_selectors(train_run: str, path: Path, seed: int):
    from .embed import build_dataset
    from .eval_offline import split_games
    from . import models as M
    ds = build_dataset(train_run)
    train_g, val_g, _ = split_games(ds, seed, fractions=(0.85, 0.15, 0.0))
    tr, va = M.tensors(ds.subset(np.isin(ds.game, train_g))), M.tensors(ds.subset(np.isin(ds.game, val_g)))
    t0 = time.time()
    models = {"predictor": M.fit(M.Predictor(), M.predictor_loss, tr, va, seed)}
    print(f"predictor trained ({time.time() - t0:.0f}s)", flush=True)
    # as in train_all: the value head is trained on the predictor's own outputs
    tr_p = dict(tr, outcome=M.predict_outcomes(models["predictor"], tr))
    va_p = dict(va, outcome=M.predict_outcomes(models["predictor"], va))
    models["value_pred_rank"] = M.fit(M.ValueHead(extra=M.N_RANKS), M.value_loss, tr_p, va_p, seed)
    models["decision_rank"] = M.fit(M.DecisionModel(extra=M.N_RANKS), M.decision_loss, tr, va, seed)
    print(f"value head and decision model trained ({time.time() - t0:.0f}s)", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: models[k].cpu().state_dict() for k in SAVED}, path)
    print(f"trained selectors on {len(train_g)} games ({len(val_g)} for validation), saved to {path}", flush=True)


class Selector:
    """Scores candidates on the CPU with the saved models and a frozen encoder."""

    def __init__(self, path: Path):
        from sentence_transformers import SentenceTransformer
        from .embed import ENCODER
        from .models import N_RANKS, DecisionModel, Predictor, ValueHead
        self.n_ranks = N_RANKS
        sd = torch.load(path, map_location="cpu")
        self.predictor = Predictor()
        self.value = ValueHead(extra=N_RANKS)
        self.decision = DecisionModel(extra=N_RANKS)
        for m, k in ((self.predictor, "predictor"), (self.value, "value_pred_rank"), (self.decision, "decision_rank")):
            m.load_state_dict(sd[k])
            m.eval()
        self.encoder = SentenceTransformer(ENCODER, device="cpu")
        self.cache: dict[str, np.ndarray] = {}

    def embed(self, texts: list[str]) -> np.ndarray:
        new = [t for t in dict.fromkeys(texts) if t not in self.cache]
        if new:
            vecs = self.encoder.encode(new, normalize_embeddings=True, show_progress_bar=False,
                                       convert_to_numpy=True).astype(np.float32)
            self.cache.update(zip(new, vecs))
        return np.stack([self.cache[t] for t in texts])

    @torch.no_grad()
    def scores(self, goal: str, observation: str, history: list[tuple[str, str]], cands: list[str]) -> dict:
        """Index of the chosen candidate for each learned arm."""
        lines = [f"{a} -> {o}" for a, o in history[-HISTORY_STEPS:]]
        g, o = self.embed([goal, observation])
        h = self.embed(lines).mean(0) if lines else np.zeros_like(g)
        n = len(cands)
        rep = lambda v: torch.from_numpy(np.tile(v, (n, 1)))
        state = torch.cat([rep(g), rep(o), rep(h)], -1)
        action = torch.from_numpy(self.embed(cands))
        rank = torch.eye(self.n_ranks)[torch.arange(n).clamp(max=self.n_ranks - 1)]
        dec = F.softmax(self.decision(state, action, rank)[0], -1)[:, 0]
        val = F.softmax(self.value(self.predictor(state, action), rep(g), rank), -1)
        hyb = dec + 10.0 * (val.argmax(1) == 0)
        # argmax returns the first maximum, so ties go to the proposer's higher-ranked candidate
        return {"decision_rank": int(dec.argmax()), "jepa_rank": int(val[:, 0].argmax()),
                "hybrid_rank": int(hyb.argmax())}


def run_arm(game: Game, arm: str, proposer, selector: Selector, cache: dict, max_steps: int) -> dict:
    env = AlfEnv(game)
    obs = env.reset()
    room = obs.feedback
    history: list[tuple[str, str]] = []
    row = {"game_id": game.game_id, "task_type": game.task_type, "arm": arm, "steps": 0, "errors": 0,
           "useful": 0, "new_llm_calls": 0, "not_first": 0, "select_s": 0.0, "actions": []}
    for _ in range(max_steps):
        if obs.won:
            break
        key = build_user_prompt(env.goal, history, obs.feedback, room)
        if key not in cache:
            cache[key] = proposer.propose(env.goal, history, obs.feedback, room=room,
                                          expert_action=obs.next_expert_action, admissible=obs.admissible)
            row["new_llm_calls"] += 1
        cands = cache[key]
        pick = 0
        if arm != "llm_first":
            t = time.time()
            pick = selector.scores(env.goal, obs.feedback, history, cands)[arm]
            row["select_s"] += time.time() - t
        out = env.step(cands[pick])
        history.append((out.action, out.feedback))
        obs = env.obs
        row["steps"] += 1
        row["errors"] += out.label == ERROR
        row["useful"] += out.label == USEFUL
        row["not_first"] += pick != 0
        row["actions"].append(out.action)
    env.close()
    row["won"] = bool(obs.won)
    row["select_s"] = round(row["select_s"], 3)
    return row


_cfg: dict = {}


def _init_worker(cfg):
    torch.set_num_threads(2)
    _cfg.update(cfg)
    _cfg["selector"] = Selector(Path(cfg["models_path"]))
    if cfg["stub"]:
        _cfg["proposer"] = StubProposer(n=cfg["n"], seed=cfg["seed"] + mp.current_process().pid)
    elif cfg["provider"] == "anthropic":
        _cfg["proposer"] = AnthropicProposer(model=cfg["model"], n=cfg["n"], effort=cfg["effort"],
                                             thinking=cfg["thinking"])
    else:
        _cfg["proposer"] = LLMProposer(model=cfg["model"], n=cfg["n"], temperature=cfg["temperature"],
                                       base_url=cfg["base_url"], thinking=cfg["thinking"],
                                       min_interval=60 * cfg["workers"] / cfg["rpm"] if cfg["rpm"] else 0,
                                       api_key=os.environ.get(cfg["api_key_env"]) if cfg["api_key_env"] else None)


def _work(game: Game):
    cache: dict = {}
    try:
        return game.game_id, [run_arm(game, arm, _cfg["proposer"], _cfg["selector"], cache, _cfg["max_steps"])
                              for arm in ARMS], None
    except Exception as e:
        return game.game_id, [], f"{type(e).__name__}: {e}"


def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def boot(values: np.ndarray, n: int = 5000, seed: int = 0) -> tuple[float, float, float]:
    """Mean with a 95% interval, resampling games."""
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(n, len(values)))].mean(1)
    return float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize(rows: list[dict], max_steps: int) -> tuple[str, dict]:
    games = sorted({r["game_id"] for r in rows})
    if not games:
        return "no games logged", {}
    by = {(r["game_id"], r["arm"]): r for r in rows}
    col = lambda arm, f: np.array([f(by[g, arm]) for g in games], dtype=float)
    won = {a: col(a, lambda r: r["won"]) for a in ARMS}
    res = {"n_games": len(games), "arms": {}, "by_task_type": {}}
    lines = [f"{len(games)} games",
             f"{'arm':15s} {'success':>8s}  {'95% interval':>14s}  {'vs llm_first':>22s}  {'steps':>6s}  "
             f"{'rejected':>8s}  {'useful':>7s}  {'overrides':>9s}"]
    for a in ARMS:
        m, lo, hi = boot(won[a])
        d, dlo, dhi = boot(won[a] - won["llm_first"])
        steps = col(a, lambda r: r["steps"])
        length = col(a, lambda r: r["steps"] if r["won"] else max_steps)
        rej = col(a, lambda r: r["errors"]).sum() / steps.sum()
        use = col(a, lambda r: r["useful"]).sum() / steps.sum()
        ovr = col(a, lambda r: r["not_first"]).sum() / steps.sum()
        res["arms"][a] = {"success": m, "success_ci": [lo, hi], "vs_llm": d, "vs_llm_ci": [dlo, dhi],
                          "mean_steps": float(length.mean()), "rejected_rate": float(rej),
                          "useful_rate": float(use), "override_rate": float(ovr),
                          "new_llm_calls": int(col(a, lambda r: r["new_llm_calls"]).sum()),
                          "select_ms_per_step": float(1000 * col(a, lambda r: r["select_s"]).sum() / steps.sum())}
        lines.append(f"{a:15s} {m:8.1%}  [{lo:5.1%}, {hi:5.1%}]  {d:+6.1%} [{dlo:+5.1%}, {dhi:+5.1%}]  "
                     f"{length.mean():6.1f}  {rej:8.1%}  {use:7.1%}  {ovr:9.1%}")
    d, dlo, dhi = boot(won["jepa_rank"] - won["decision_rank"])
    res["jepa_rank_minus_decision_rank"] = {"diff": d, "ci": [dlo, dhi]}
    lines.append(f"jepa_rank minus decision_rank: {d:+.1%} [{dlo:+.1%}, {dhi:+.1%}]")
    lines.append("steps = mean episode length, counting a failed game as the step limit")
    lines.append(f"\n{'task type':32s} games  " + "  ".join(f"{a:>13s}" for a in ARMS))
    types = np.array([by[g, ARMS[0]]["task_type"] for g in games])
    for tt in TASK_TYPES:
        mask = types == tt
        if mask.any():
            res["by_task_type"][tt] = {"n_games": int(mask.sum()), **{a: float(won[a][mask].mean()) for a in ARMS}}
            lines.append(f"{tt:32s} {mask.sum():5d}  " + "  ".join(f"{won[a][mask].mean():13.0%}" for a in ARMS))
    return "\n".join(lines), res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="name of this online run; logs go to data/logs/RUN/")
    ap.add_argument("--train-run", required=True, help="Phase 1 run the selectors are trained on")
    ap.add_argument("--split", default="valid_unseen")
    ap.add_argument("--games", type=int, default=0, help="limit the number of games (0 = all)")
    ap.add_argument("--provider", choices=("anthropic", "openai"), default="anthropic")
    ap.add_argument("--model")
    ap.add_argument("--base-url")
    ap.add_argument("--api-key-env")
    ap.add_argument("--effort", default="low")
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--rpm", type=float, default=0)
    ap.add_argument("--stub", action="store_true", help="pipeline test with no LLM")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-steps", type=int, default=40)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    log_path = LOG_ROOT / args.run / "online.jsonl"
    res_dir = RESULTS / args.run
    if not args.summary:
        if not args.stub and not args.model:
            if args.provider != "anthropic":
                ap.error("give --model, or --stub for a pipeline test")
            args.model = "claude-haiku-5-5"
        models_path = res_dir / "models.pt"
        if not models_path.exists():
            train_selectors(args.train_run, models_path, args.seed)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        cfg = {k: getattr(args, k) for k in ("train_run", "split", "provider", "model", "base_url", "api_key_env",
                                             "effort", "thinking", "rpm", "workers", "stub", "n", "temperature",
                                             "max_steps", "seed")}
        cfg["models_path"] = str(models_path)
        (log_path.parent / "online_config.json").write_text(json.dumps(cfg, indent=2))

        done = {r["game_id"] for r in load_rows(log_path)}
        games = list_games(args.split)
        if args.games:
            # spread a limited run over the task types
            games = [g for i in range(len(games)) for g in
                     [gs[i] for gs in ([g for g in games if g.task_type == tt] for tt in TASK_TYPES) if i < len(gs)]
                     ][:args.games]
        games = [g for g in games if g.game_id not in done]
        print(f"{len(done)} games already logged, {len(games)} to run, {args.workers} workers", flush=True)
        t0, failed, calls = time.time(), 0, 0
        with mp.Pool(args.workers, initializer=_init_worker, initargs=(cfg,)) as pool, open(log_path, "a") as f:
            for i, (gid, rows, err) in enumerate(pool.imap_unordered(_work, games, chunksize=1), 1):
                if err:
                    failed += 1
                    print(f"FAILED {gid}: {err}", flush=True)
                    continue
                for r in rows:
                    f.write(json.dumps(r) + "\n")
                f.flush()
                calls += sum(r["new_llm_calls"] for r in rows)
                if i % 5 == 0 or i == len(games):
                    print(f"{i}/{len(games)} games, {time.time() - t0:.0f}s, {calls} LLM calls", flush=True)
        print(f"\ndone: {failed} games failed\n")

    text, res = summarize(load_rows(log_path), args.max_steps)
    print(text)
    if res:
        res_dir.mkdir(parents=True, exist_ok=True)
        (res_dir / "online.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
