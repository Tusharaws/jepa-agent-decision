"""Phase 0 checks. Run: uv run python -m src.phase0_check [games_per_type] [workers]

For a sample of training games of every task type:

1. Achievable: from the start and from states reached by random actions, following
   `next_expert_action` lowers steps-to-go by exactly one each step and wins. So a
   plan of that length exists.
2. Lower bound: no action taken anywhere lowers steps-to-go by more than one. So no
   shorter plan was missed (on the transitions sampled).
3. Planner agreement: where TextWorld's planner answers quickly, its plan length at
   the start equals ours.
4. Determinism: the same action sequence gives the same observations twice.
5. Branching: `branch` agrees with actually taking the action.
6. Timing and label mix under a noisy policy.
"""

from __future__ import annotations

import multiprocessing as mp
import random
import sys
import time
from collections import Counter

from .env import AlfEnv, TASK_TYPES, _start, list_games

NOISY_STEPS = 30
PLANNER_TIMEOUT = 20
GARBAGE = ["open fridge 9", "take sword from table", "fly to the moon", "heat nothing with microwave 1"]


def follow_expert(env: AlfEnv, limit: int = 40) -> str | None:
    """Follow next_expert_action to the end. Returns a problem description, or None."""
    for _ in range(limit):
        obs = env.obs
        if obs.won:
            return None if obs.steps_to_go == 0 else "won but steps_to_go != 0"
        if obs.steps_to_go is None:
            return "no plan found"
        if obs.steps_to_go == 0:
            return "steps_to_go == 0 but not won"
        if obs.next_expert_action not in obs.admissible:
            return f"expert action not admissible: {obs.next_expert_action!r}"
        out = env.step(obs.next_expert_action)
        if out.steps_after != out.steps_before - 1:
            return f"{out.action!r}: steps {out.steps_before}->{out.steps_after}"
    return "did not finish"


def check_game(game):
    rng = random.Random(hash(game.game_id) % 10_000)
    res = {"game": game, "problems": [], "labels": Counter(), "transitions": 0}
    env = AlfEnv(game)

    # 1. Achievable from the start.
    start = env.reset()
    res["start_steps"] = start.steps_to_go
    p = follow_expert(env)
    if p:
        res["problems"].append(f"expert from start: {p}")

    # Noisy walk: expert, random admissible and invalid commands mixed.
    obs = env.reset()
    actions, trace, outs = [], [(obs.feedback, obs.steps_to_go)], []
    t0 = time.time()
    for _ in range(NOISY_STEPS):
        if obs.won:
            break
        r = rng.random()
        if r < 0.3 and obs.next_expert_action:
            a = obs.next_expert_action
        elif r < 0.85:
            a = rng.choice(obs.admissible)
        else:
            a = rng.choice(GARBAGE)
        out = env.step(a)
        actions.append(a)
        outs.append(out)
        trace.append((out.feedback, out.steps_after))
        obs = env.obs
        # 2. Lower bound.
        if out.steps_before is not None and out.steps_after is not None and out.steps_after < out.steps_before - 1:
            res["problems"].append(f"dropped by >1: {a!r} {out.steps_before}->{out.steps_after}")
        res["labels"][out.label] += 1
    res["step_ms"] = (time.time() - t0) / max(1, len(actions)) * 1000
    res["transitions"] = len(actions)

    # 1 again: achievable from wherever the noisy walk ended.
    p = follow_expert(env)
    if p:
        res["problems"].append(f"expert after noisy walk: {p}")

    # 4. Determinism.
    obs = env.reset()
    trace2 = [(obs.feedback, obs.steps_to_go)]
    for a in actions:
        o = env.step(a)
        trace2.append((o.feedback, o.steps_after))
    if trace != trace2:
        res["problems"].append("not deterministic under replay")

    # 5. Branching, at up to 6 points along the walk.
    obs = env.reset()
    check_at = set(rng.sample(range(len(actions)), min(6, len(actions))))
    t_branch, n_branch = 0.0, 0
    for i, a in enumerate(actions):
        if i in check_at:
            cands = [rng.choice(GARBAGE), a] + rng.sample(obs.admissible, min(3, len(obs.admissible)))
            t = time.time()
            b = env.branch(cands)[1]
            t_branch += time.time() - t
            n_branch += 1
            if (b.feedback, b.steps_after, b.label) != (outs[i].feedback, outs[i].steps_after, outs[i].label):
                res["problems"].append(f"branch mismatch at step {i}: {a!r}")
        env.step(a)
        obs = env.obs
    res["branch_ms"] = t_branch / max(1, n_branch) * 1000
    env.close()
    return res


def _planner_len(path, q):
    env = _start(path, planner=True)
    q.put(len(env.reset()["policy_commands"]))


def planner_len(game) -> int | None:
    """Start-state plan length from TextWorld's planner, or None if it is too slow."""
    q = mp.Queue()
    proc = mp.Process(target=_planner_len, args=(game.path, q))
    proc.start()
    proc.join(PLANNER_TIMEOUT)
    if proc.is_alive():
        proc.kill()
        proc.join()
        return None
    return q.get() if not q.empty() else None


def check_game_with_planner(game):
    res = check_game(game)
    res["planner"] = planner_len(game)
    return res


def main():
    per_type = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    rng = random.Random(0)
    games = list_games("train")
    chosen = [g for tt in TASK_TYPES for g in rng.sample([g for g in games if g.task_type == tt], per_type)]
    print(f"{len(chosen)} games, {per_type} per task type, {workers} workers", flush=True)

    from concurrent.futures import ThreadPoolExecutor
    t0 = time.time()
    with mp.Pool(workers) as pool:
        results = pool.map(check_game, chosen, chunksize=1)
    print(f"checks done in {time.time() - t0:.0f}s; asking the planner...", flush=True)
    with ThreadPoolExecutor(workers) as ex:
        planner = list(ex.map(planner_len, chosen))

    print(f"\n{'task type':32s} games  ok  planner agrees  median steps  step ms  branch ms")
    for tt in TASK_TYPES:
        rows = [(r, p) for r, p in zip(results, planner) if r["game"].task_type == tt]
        ok = sum(not r["problems"] for r, _ in rows)
        answered = [(r, p) for r, p in rows if p is not None]
        agree = sum(r["start_steps"] == p for r, p in answered)
        med = lambda k: sorted(r[k] for r, _ in rows)[len(rows) // 2]
        print(f"{tt:32s} {len(rows):5d} {ok:3d}  {agree:3d} of {len(answered):3d}     "
              f"{med('start_steps'):6}      {med('step_ms'):5.0f}  {med('branch_ms'):7.0f}")

    for r, p in zip(results, planner):
        for prob in r["problems"]:
            print(f"PROBLEM {r['game'].game_id}: {prob}")
        if p is not None and p != r["start_steps"]:
            print(f"PLANNER DIFFERS {r['game'].game_id}: ours {r['start_steps']}, planner {p}")

    labels = sum((r["labels"] for r in results), Counter())
    total = sum(labels.values())
    print(f"\nGames with no problems:  {sum(not r['problems'] for r in results)}/{len(results)}")
    print(f"Transitions checked:     {sum(r['transitions'] for r in results)}")
    print("Label mix (noisy walk):  " + ", ".join(f"{k} {v / total:.0%}" for k, v in labels.most_common()))


if __name__ == "__main__":
    main()
