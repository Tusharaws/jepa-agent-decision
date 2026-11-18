"""Phase 5: does 2-step lookahead through the world model pick better actions?

    uv run python -m src.lookahead --run NAME [--seeds 5]

Offline, on the Phase 1 log, with the in-distribution split of eval_offline.

For candidate i of a step, the predictor gives the outcome embedding o_i. The
imagined next state is the current state with o_i as the observation. Every
other candidate j of the same step is then scored as a second action from that
imagined state, and

    score(i) = one_step(i) + lam * max_j second_step(i, j)

lam is chosen per seed on the validation games from LAMS (0 = no lookahead).

Selectors:
    jepa_rank / decision_rank    one-step baselines (as in eval_offline)
    jepa_look        JEPA one-step score + value head on the predicted second outcome
    decision_look    decision one-step score + decision model on the imagined state
    *_look_true      the same, but imagining with the true first outcome instead of
                     the predicted one (not deployable; shows what a perfect predictor gives)

Second-step scores use the models without the rank input, because the
proposer's ranking was made for the current state, not the imagined one.
Results are written to results/NAME/lookahead.json.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import models as M
from .embed import Dataset, build_dataset
from .eval_offline import RESULTS, USEFUL, boot, split_games

LAMS = (0.0, 0.1, 0.25, 0.5, 1.0)
NAMES = ("llm_first", "decision_rank", "jepa_rank", "decision_look", "jepa_look",
         "decision_look_true", "jepa_look_true")


def pairs(ds: Dataset) -> tuple[np.ndarray, np.ndarray]:
    """Row indices (i, j) for every ordered pair of different candidates of the same step."""
    order = np.argsort(ds.step_id, kind="stable")
    _, start, count = np.unique(ds.step_id[order], return_index=True, return_counts=True)
    ii, jj = [], []
    for s, c in zip(start, count):
        idx = order[s:s + c]
        a, b = np.meshgrid(idx, idx, indexing="ij")
        keep = a != b
        ii.append(a[keep])
        jj.append(b[keep])
    return np.concatenate(ii), np.concatenate(jj)


@torch.no_grad()
def lookahead_scores(models: dict, ds: Dataset) -> dict[str, np.ndarray]:
    """One-step scores and the best second-step score for each candidate row."""
    t = M.tensors(ds)
    dim = M.DIM
    p_use = lambda logits: F.softmax(logits, -1)[:, USEFUL]
    pred = M.predict_outcomes(models["predictor"], t)
    one = {"decision": p_use(models["decision_rank"](t["state"], t["action"], t["rank"])[0]),
           "jepa": p_use(models["value_pred_rank"](pred, t["goal"], t["rank"]))}
    ii, jj = pairs(ds)
    ii_t, jj_t = torch.from_numpy(ii).to(M.DEVICE), torch.from_numpy(jj).to(M.DEVICE)
    out = {"one_decision": one["decision"].cpu().numpy(), "one_jepa": one["jepa"].cpu().numpy()}
    for tag, first_outcome in (("", pred), ("_true", t["outcome"])):
        best = {"decision": torch.zeros(len(ds), device=M.DEVICE), "jepa": torch.zeros(len(ds), device=M.DEVICE)}
        for k in range(0, len(ii), 8192):
            i, j = ii_t[k:k + 8192], jj_t[k:k + 8192]
            # imagined state: same goal and history, observation replaced by the first action's outcome
            state2 = torch.cat([t["state"][i, :dim], first_outcome[i], t["state"][i, 2 * dim:]], -1)
            q = {"decision": p_use(models["decision"](state2, t["action"][j])[0]),
                 "jepa": p_use(models["value_pred"](models["predictor"](state2, t["action"][j]), t["goal"][i]))}
            for name in best:
                best[name].scatter_reduce_(0, i, q[name], reduce="amax", include_self=True)
        out["two_decision" + tag] = best["decision"].cpu().numpy()
        out["two_jepa" + tag] = best["jepa"].cpu().numpy()
    return out


def picks(score: np.ndarray, ds: Dataset) -> np.ndarray:
    """Per step (in step-id order): is the top-scored candidate useful. Ties go to the higher rank."""
    order = np.lexsort((ds.rank, ds.step_id))
    _, start, count = np.unique(ds.step_id[order], return_index=True, return_counts=True)
    return np.array([ds.label[order[s:s + c]][np.argmax(score[order[s:s + c]])] == USEFUL
                     for s, c in zip(start, count)], dtype=float)


def step_games(ds: Dataset) -> np.ndarray:
    _, first = np.unique(ds.step_id, return_index=True)
    return ds.game[first]


def train(train_ds: Dataset, val_ds: Dataset, seed: int) -> dict:
    tr, va = M.tensors(train_ds), M.tensors(val_ds)
    m = {"predictor": M.fit(M.Predictor(), M.predictor_loss, tr, va, seed)}
    tr_p = dict(tr, outcome=M.predict_outcomes(m["predictor"], tr))
    va_p = dict(va, outcome=M.predict_outcomes(m["predictor"], va))
    m["value_pred"] = M.fit(M.ValueHead(), M.value_loss, tr_p, va_p, seed)
    m["value_pred_rank"] = M.fit(M.ValueHead(extra=M.N_RANKS), M.value_loss, tr_p, va_p, seed)
    m["decision"] = M.fit(M.DecisionModel(), M.decision_loss, tr, va, seed)
    m["decision_rank"] = M.fit(M.DecisionModel(extra=M.N_RANKS), M.decision_loss, tr, va, seed)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--split-seed", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    ds = build_dataset(args.run)
    train_g, val_g, test_g = split_games(ds, args.split_seed)
    sub = lambda gs: ds.subset(np.isin(ds.game, gs))
    train_ds, val_ds, test_ds = sub(train_g), sub(val_g), sub(test_g)
    games = step_games(test_ds)
    acc = {n: np.zeros(len(games)) for n in NAMES}
    lams = {n: [] for n in NAMES if "look" in n}
    for seed in range(args.seeds):
        models = train(train_ds, val_ds, seed)
        sv, st = lookahead_scores(models, val_ds), lookahead_scores(models, test_ds)
        acc["llm_first"] += picks(-test_ds.rank.astype(float), test_ds)
        for arm in ("decision", "jepa"):
            acc[f"{arm}_rank"] += picks(st[f"one_{arm}"], test_ds)
            for tag in ("", "_true"):
                val_use = [picks(sv[f"one_{arm}"] + lam * sv[f"two_{arm}{tag}"], val_ds).mean() for lam in LAMS]
                lam = LAMS[int(np.argmax(val_use))]  # first maximum, so no lookahead wins ties
                lams[f"{arm}_look{tag}"].append(lam)
                acc[f"{arm}_look{tag}"] += picks(st[f"one_{arm}"] + lam * st[f"two_{arm}{tag}"], test_ds)
        print(f"seed {seed} done ({time.time() - t0:.0f}s); lam chosen: "
              + ", ".join(f"{k} {v[-1]}" for k, v in lams.items()), flush=True)
    for n in acc:
        acc[n] /= args.seeds

    res = {"run": args.run, "seeds": args.seeds, "n_test_games": int(len(np.unique(games))),
           "n_test_steps": int(len(games)), "lam_chosen": lams, "selectors": {}, "differences": {}}
    print(f"\nLOOKAHEAD, in-distribution  ({res['n_test_games']} test games, {res['n_test_steps']} steps)")
    print(f"  {'selector':20s} {'useful':>7s}  {'95% interval':>14s}")
    for n in NAMES:
        m, lo, hi = boot(acc[n], games)
        res["selectors"][n] = {"useful": m, "useful_ci": [lo, hi]}
        print(f"  {n:20s} {m:7.1%}  [{lo:5.1%}, {hi:5.1%}]")
    for a, b in (("jepa_look", "jepa_rank"), ("decision_look", "decision_rank"), ("jepa_look", "decision_rank"),
                 ("jepa_look_true", "jepa_rank"), ("decision_look_true", "decision_rank")):
        d, lo, hi = boot(acc[a] - acc[b], games)
        res["differences"][f"{a}_minus_{b}"] = {"diff": d, "ci": [lo, hi]}
        print(f"  {a} minus {b}: {d:+.1%} [{lo:+.1%}, {hi:+.1%}]")
    path = RESULTS / args.run / "lookahead.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res, indent=2))
    print(f"\nwrote {path}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
