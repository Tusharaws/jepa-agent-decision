"""Phases 2 and 3: train the selectors and compare them offline.

    uv run python -m src.eval_offline --run NAME [--seeds 5] [--skip-generalisation]

For each held-out step, each selector picks one of the logged candidates; the
candidate's true label is known from Phase 1. Reported per selector:

    useful   share of steps where the chosen candidate is useful
    error    share where the game rejects it
    gap      share of the distance from the proposer's own first choice to the
             oracle that the selector closes

Selectors ending in _rank also see the proposer's ranking of each candidate.

Splits (always by game, never by step):
    in-distribution   70 / 15 / 15 train / validation / test, stratified by task type
    held-out type     train on five task types, test on the sixth; all six in turn
    held-out verb     drop every candidate with one verb from training

Intervals are 95% bootstrap intervals over test games, with model seeds averaged.
Results are written to results/NAME/offline.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .embed import Dataset, build_dataset
from .env import LABELS, TASK_TYPES
from .models import DecisionModel, Predictor, ValueHead, n_params, score_all, train_all

RESULTS = Path(__file__).resolve().parent.parent / "results"
USEFUL, ERROR = LABELS.index("useful"), LABELS.index("error")
SELECTORS = ["llm_first", "decision", "jepa", "hybrid", "decision_rank", "jepa_rank", "hybrid_rank",
             "decision_aux", "jepa_true_trained", "value_on_true_outcome", "random", "oracle"]
HELD_OUT_VERBS = ("cool", "heat", "clean", "use")


def split_games(ds: Dataset, seed: int, fractions=(0.7, 0.15, 0.15)):
    """Game-index arrays for train / val / test, stratified by task type."""
    rng = np.random.default_rng(seed)
    type_of = {g: tt for g, tt in zip(ds.game, ds.task_type)}
    parts = ([], [], [])
    for tt in TASK_TYPES:
        gs = np.array(sorted(g for g, t in type_of.items() if t == tt))
        rng.shuffle(gs)
        a = int(round(fractions[0] * len(gs)))
        b = a + max(1, int(round(fractions[1] * len(gs)))) if len(gs) > 2 else a
        for part, chunk in zip(parts, (gs[:a], gs[a:b], gs[b:])):
            part.extend(chunk.tolist())
    return tuple(np.array(p) for p in parts)


def choices(scores: dict[str, np.ndarray], ds: Dataset) -> dict[str, dict]:
    """For each selector, per-step arrays: was the chosen candidate useful / an error."""
    steps, start = np.unique(ds.step_id, return_index=True)
    order = np.argsort(ds.step_id, kind="stable")
    bounds = np.append(np.searchsorted(ds.step_id[order], steps), len(order))
    lab, rank = ds.label, ds.rank
    hybrid = scores["decision"][:, USEFUL] + 10.0 * (scores["jepa"].argmax(1) == USEFUL)
    per_cand = {
        "llm_first": -rank.astype(float),
        "decision": scores["decision"][:, USEFUL],
        "jepa": scores["jepa"][:, USEFUL],
        "hybrid": hybrid,
        "decision_rank": scores["decision_rank"][:, USEFUL],
        "jepa_rank": scores["jepa_rank"][:, USEFUL],
        "hybrid_rank": scores["decision_rank"][:, USEFUL] + 10.0 * (scores["jepa_rank"].argmax(1) == USEFUL),
        "decision_aux": scores["decision_aux"][:, USEFUL],
        "jepa_true_trained": scores["jepa_true_trained"][:, USEFUL],
        "value_on_true_outcome": scores["value_on_true_outcome"][:, USEFUL],
    }
    out = {k: {"useful": np.zeros(len(steps)), "error": np.zeros(len(steps))} for k in SELECTORS}
    for i in range(len(steps)):
        idx = order[bounds[i]:bounds[i + 1]]
        idx = idx[np.argsort(rank[idx], kind="stable")]  # ties go to the proposer's higher-ranked candidate
        for name, s in per_cand.items():
            pick = idx[np.argmax(s[idx])]
            out[name]["useful"][i] = lab[pick] == USEFUL
            out[name]["error"][i] = lab[pick] == ERROR
        out["random"]["useful"][i] = (lab[idx] == USEFUL).mean()
        out["random"]["error"][i] = (lab[idx] == ERROR).mean()
        out["oracle"]["useful"][i] = (lab[idx] == USEFUL).any()
        out["oracle"]["error"][i] = (lab[idx] == ERROR).all()
    step_game = ds.game[order[bounds[:-1]]]
    return {"per_step": out, "step_game": step_game, "steps": steps}


def boot(values: np.ndarray, groups: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Mean with a 95% interval, resampling groups (games)."""
    rng = np.random.default_rng(seed)
    ids = np.unique(groups)
    sums = np.array([values[groups == g].sum() for g in ids])
    cnts = np.array([(groups == g).sum() for g in ids])
    pick = rng.integers(0, len(ids), size=(n, len(ids)))
    means = sums[pick].sum(1) / cnts[pick].sum(1)
    return float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def auc(score: np.ndarray, positive: np.ndarray) -> float | None:
    pos, neg = score[positive], score[~positive]
    if len(pos) == 0 or len(neg) == 0:
        return None
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def evaluate(ds: Dataset, train_g, val_g, test_g, seeds: int, train_mask=None, step_filter=None) -> dict:
    """Train on train_g (optionally a row mask within it), report on test_g."""
    in_g = lambda gs: np.isin(ds.game, gs)
    tr_mask = in_g(train_g) if train_mask is None else in_g(train_g) & train_mask
    va_mask = in_g(val_g) if train_mask is None else in_g(val_g) & train_mask
    train, val, test = ds.subset(tr_mask), ds.subset(va_mask), ds.subset(in_g(test_g))
    if len(val) == 0:  # only happens on tiny smoke-test datasets
        val = train
    acc, cos, labels = None, [], test.label
    for seed in range(seeds):
        scores = score_all(train_all(train, val, seed), test)
        c = choices(scores, test)
        cos.append(scores.pop("_pred_cos"))
        if acc is None:
            acc = {k: {m: v.copy() for m, v in d.items()} for k, d in c["per_step"].items()}
            sums = {k: v.copy() for k, v in scores.items()}
        else:
            for k, d in c["per_step"].items():
                for m in d:
                    acc[k][m] += d[m]
            for k in sums:
                sums[k] += scores[k]
    for k in acc:
        for m in acc[k]:
            acc[k][m] /= seeds
    step_game, steps = c["step_game"], c["steps"]
    keep = np.ones(len(steps), bool) if step_filter is None else step_filter(test, steps)
    res = {"n_train_rows": int(len(train)), "n_test_steps": int(keep.sum()),
           "n_test_games": int(len(np.unique(step_game[keep]))), "selectors": {}}
    if keep.sum() == 0:
        return res
    base, top = acc["llm_first"]["useful"][keep], acc["oracle"]["useful"][keep]
    for name in SELECTORS:
        u, e = acc[name]["useful"][keep], acc[name]["error"][keep]
        mean, lo, hi = boot(u, step_game[keep])
        d_mean, d_lo, d_hi = boot(u - base, step_game[keep])
        denom = top.mean() - base.mean()
        res["selectors"][name] = {
            "useful": mean, "useful_ci": [lo, hi], "error": float(e.mean()),
            "vs_llm": d_mean, "vs_llm_ci": [d_lo, d_hi],
            "gap_closed": float((mean - base.mean()) / denom) if denom > 1e-9 else None,
        }
    d_mean, d_lo, d_hi = boot(acc["jepa"]["useful"][keep] - acc["decision"]["useful"][keep], step_game[keep])
    res["jepa_minus_decision"] = {"diff": d_mean, "ci": [d_lo, d_hi]}
    d_mean, d_lo, d_hi = boot(acc["jepa_rank"]["useful"][keep] - acc["decision_rank"]["useful"][keep], step_game[keep])
    res["jepa_rank_minus_decision_rank"] = {"diff": d_mean, "ci": [d_lo, d_hi]}
    mean_cos = np.mean(cos, 0)
    res["predictor_cosine"] = {LABELS[i]: float(mean_cos[labels == i].mean()) for i in range(3) if (labels == i).any()}
    res["_scores"] = {k: v / seeds for k, v in sums.items()}
    res["_test"] = test
    return res


def table(res: dict, title: str) -> str:
    lines = [f"\n{title}  ({res['n_test_games']} test games, {res['n_test_steps']} steps, "
             f"{res['n_train_rows']} training rows)",
             f"  {'selector':24s} {'useful':>7s}  {'95% interval':>14s}  {'error':>6s}  {'vs first choice':>22s}  {'gap closed':>10s}"]
    for name, r in res.get("selectors", {}).items():
        gap = f"{r['gap_closed']:.0%}" if r["gap_closed"] is not None else "n/a"
        lines.append(f"  {name:24s} {r['useful']:7.1%}  [{r['useful_ci'][0]:5.1%}, {r['useful_ci'][1]:5.1%}]  "
                     f"{r['error']:6.1%}  {r['vs_llm']:+6.1%} [{r['vs_llm_ci'][0]:+5.1%}, {r['vs_llm_ci'][1]:+5.1%}]  {gap:>10s}")
    if "jepa_minus_decision" in res:
        j = res["jepa_minus_decision"]
        lines.append(f"  jepa minus decision: {j['diff']:+.1%} [{j['ci'][0]:+.1%}, {j['ci'][1]:+.1%}]")
        j = res["jepa_rank_minus_decision_rank"]
        lines.append(f"  jepa_rank minus decision_rank: {j['diff']:+.1%} [{j['ci'][0]:+.1%}, {j['ci'][1]:+.1%}]")
    if "predictor_cosine" in res:
        lines.append("  predictor cosine to true outcome: "
                     + ", ".join(f"{k} {v:.2f}" for k, v in res["predictor_cosine"].items()))
    return "\n".join(lines)


def strip(res: dict) -> dict:
    return {k: v for k, v in res.items() if not k.startswith("_")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--skip-generalisation", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    ds = build_dataset(args.run)
    jepa_p = n_params(Predictor(), ValueHead())
    print(f"{len(ds)} rows, {len(np.unique(ds.step_id))} steps, {len(ds.games)} games")
    print(f"parameters: JEPA arm (predictor + value head) {jepa_p / 1e6:.2f}M, "
          f"decision model {n_params(DecisionModel()) / 1e6:.2f}M")
    out = {"run": args.run, "seeds": args.seeds}

    train_g, val_g, test_g = split_games(ds, args.split_seed)
    res = evaluate(ds, train_g, val_g, test_g, args.seeds)
    print(table(res, "IN-DISTRIBUTION"), flush=True)
    out["in_distribution"] = strip(res)

    if not args.skip_generalisation:
        # Held-out verb: reuse the in-distribution game split; hide the verb's rows from training.
        out["held_out_verb"] = {}
        for verb in HELD_OUT_VERBS:
            n_rows = int((ds.verb == verb).sum())
            if n_rows < 30:
                continue
            has_verb = lambda test, steps, v=verb: np.array(
                [(test.verb[test.step_id == s] == v).any() for s in steps])
            r = evaluate(ds, train_g, val_g, test_g, args.seeds, train_mask=ds.verb != verb, step_filter=has_verb)
            if r.get("selectors"):
                test, rows = r["_test"], r["_test"].verb == verb
                r["auc_useful_on_verb_rows"] = {k: auc(v[rows, USEFUL], test.label[rows] == USEFUL)
                                                for k, v in r["_scores"].items()}
                print(table(r, f"HELD-OUT VERB '{verb}' (steps with a '{verb}' candidate; {n_rows} such rows overall)"))
                a = r["auc_useful_on_verb_rows"]
                print("  AUC for 'useful' on the held-out verb's own rows: "
                      + ", ".join(f"{k} {v:.2f}" for k, v in a.items() if v is not None), flush=True)
            out["held_out_verb"][verb] = strip(r)

        out["held_out_type"] = {}
        for tt in TASK_TYPES:
            rest = np.unique(ds.game[ds.task_type != tt])
            held = np.unique(ds.game[ds.task_type == tt])
            if len(held) == 0:
                continue
            rng = np.random.default_rng(args.split_seed)
            rng.shuffle(rest)
            n_val = max(1, int(0.15 * len(rest)))
            r = evaluate(ds, rest[n_val:], rest[:n_val], held, args.seeds)
            print(table(r, f"HELD-OUT TASK TYPE {tt}"), flush=True)
            out["held_out_type"][tt] = strip(r)

    path = RESULTS / args.run / "offline.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {path}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
