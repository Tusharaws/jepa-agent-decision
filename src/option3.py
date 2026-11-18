"""Option 3 on AgentProcessBench: judge a tool call by its *predicted* result.

    llama-server -m CLM-Q8_0.gguf --embedding --pooling last -c 8192 -np 2 -ngl 99 -b 4096 -ub 4096 --port 8090
    uv run python -m src.option3 --encode-only      # embed every step (resumable, cached)
    uv run python -m src.option3 [--seeds 3]        # train and compare; writes results/apb/option3.json

Every text goes through CLM-v0.1-8B's frozen Qwen3-8B encoder (4096-d, last-token
pooling, L2-normalised):

    s   the situation, followed by QUESTION        (what CLM's state head reads)
    a   the tool call                               (what CLM's action head reads)
    o   "tool call -> real tool response"           (same space, so the action head can read it)

Selectors, each giving a score where higher means "this step is correct":

    clm_action / clm_true         CLM's released heads, no training, on a / on o
    heads_action                  CLM heads fine-tuned on (s, a)           decision model
    heads_true                    CLM heads fine-tuned on (s, o)           ceiling: real result known
    heads_pred                    CLM heads fine-tuned on (s, P(s, a))     Option 3
    heads_pred_sigreg             same, predictor trained with SIGReg      Option 3, LeJEPA regulariser
    mlp_action / mlp_true         plain MLP on [s, a] / [s, o]             non-CLM reference

P is the predictor: [s, a] -> o, cosine loss, needs no labels. The Option 3
heads are trained on out-of-fold predictions, so they see predictions as
imperfect as the ones they get at test time.

Labels: +1 correct vs -1 incorrect (0 = neutral is left out). Main metric: AUC
for telling correct from incorrect steps, averaged over the four sources, with
95% bootstrap intervals over tasks. Splits: by task (all 5 agents' runs of a
task stay together), in-distribution 70/15/15 per source, and leave-one-source-out.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .apb import ROOT, SOURCES, load_steps
from .clm_eval import CLM

QUESTION = "Which tool call should the assistant make next?"
RESULTS = Path(__file__).resolve().parent.parent / "results" / "apb"
DIM = 4096
torch.set_num_threads(8)


# ---------------------------------------------------------------- data

def encode(rows: list[dict], clm: CLM) -> dict[str, np.ndarray]:
    print(f"encoding {len(rows)} steps (short texts first)", flush=True)
    a = clm.embed([r["action"] for r in rows])
    o = clm.embed([f"{r['action']} -> {r['outcome']}" for r in rows])
    s = clm.embed([f"{r['state']}\n\n{QUESTION}" for r in rows])
    clm.save()
    return {"s": s, "a": a, "o": o}


def split_tasks(rows, seed: int, holdout: str | None):
    """Task-id arrays (train, val, test). holdout=source: test on it, train on the rest."""
    rng = np.random.default_rng(seed)
    tasks = {}
    for r in rows:
        tasks.setdefault(r["source"], set()).add(r["task"])
    tr, va, te = [], [], []
    for src in SOURCES:
        ts = np.array(sorted(tasks[src]))
        rng.shuffle(ts)
        if holdout is not None:
            if src == holdout:
                te += ts.tolist()
            else:
                n_val = max(1, int(round(0.15 * len(ts))))
                va += ts[:n_val].tolist()
                tr += ts[n_val:].tolist()
            continue
        a, b = int(round(0.7 * len(ts))), int(round(0.85 * len(ts)))
        tr += ts[:a].tolist(); va += ts[a:b].tolist(); te += ts[b:].tolist()
    return set(tr), set(va), set(te)


# ---------------------------------------------------------------- models

def mlp(sizes, dropout=0.1):
    layers = []
    for i in range(len(sizes) - 2):
        layers += [nn.Linear(sizes[i], sizes[i + 1]), nn.GELU(), nn.Dropout(dropout)]
    return nn.Sequential(*layers, nn.Linear(sizes[-2], sizes[-1]))


class Predictor(nn.Module):
    """[s, a] -> o. Residual on a: "a -> result" starts with the call's own text."""
    def __init__(self, hidden=512):
        super().__init__()
        self.net = mlp([2 * DIM, hidden, hidden, DIM])

    def forward(self, s, a):
        return F.normalize(a + self.net(torch.cat([s, a], -1)), dim=-1)


def sigreg(z: torch.Tensor, n_dirs=128, t_max=3.0, n_t=17) -> torch.Tensor:
    """LeJEPA's SIGReg (Epps-Pulley form): distance between the empirical characteristic
    function of random 1-D projections of z and that of N(0, 1). z is scaled by sqrt(d)
    so that vectors spread evenly over the unit sphere score as Gaussian."""
    z = z * z.shape[1] ** 0.5
    u = F.normalize(torch.randn(z.shape[1], n_dirs, device=z.device), dim=0)
    p = z @ u                                                   # [batch, dirs]
    t = torch.linspace(0, t_max, n_t, device=z.device)
    pt = p[..., None] * t                                       # [batch, dirs, t]
    target = torch.exp(-t ** 2 / 2)
    err = (torch.cos(pt).mean(0) - target) ** 2 + torch.sin(pt).mean(0) ** 2
    return torch.trapezoid(err * target, t, dim=-1).mean() * z.shape[0]


class Heads(nn.Module):
    """CLM's state and action heads (fine-tuned) -> logit of 'correct'."""
    def __init__(self, clm: CLM):
        super().__init__()
        self.sh, self.ah = copy.deepcopy(clm.state_head), copy.deepcopy(clm.action_head)
        self.scale = clm.scale
        self.w = nn.Parameter(torch.tensor(0.2))
        self.b = nn.Parameter(torch.tensor(0.0))

    def cos(self, s, c):
        return (F.normalize(self.sh(s), dim=-1) * F.normalize(self.ah(c), dim=-1)).sum(-1)

    def forward(self, s, c):
        return self.w * self.scale * self.cos(s, c) + self.b


class MLPJudge(nn.Module):
    def __init__(self, hidden=512):
        super().__init__()
        self.net = mlp([2 * DIM, hidden, hidden, 1])

    def forward(self, s, c):
        return self.net(torch.cat([s, c], -1)).squeeze(-1)


def fit(model, loss_fn, train: dict, val: dict, seed: int, lr: float, epochs=40, patience=5, batch=64,
        param_groups=None):
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(param_groups or model.parameters(), lr=lr, weight_decay=1e-2)
    n = len(next(iter(train.values())))
    gen = torch.Generator().manual_seed(seed)
    best, best_state, bad = float("inf"), None, 0
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            loss = loss_fn(model, {k: v[idx] for k, v in train.items()})
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            v = loss_fn(model, val).item()
        if v < best - 1e-4:
            best, best_state, bad = v, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model.eval()


def pred_loss(lam):
    def f(model, b):
        p = model(b["s"], b["a"])
        loss = (1 - (p * b["o"]).sum(-1)).mean()
        return loss + lam * sigreg(p) if lam and model.training else loss
    return f


def judge_loss(key):
    def f(model, b):
        logit = model(b["s"], b[key])
        return F.binary_cross_entropy_with_logits(logit, b["y"], weight=b["wt"])
    return f


def train_predictor(T: dict, tr_idx, va_idx, seed, lam):
    sub = lambda idx: {k: T[k][idx] for k in ("s", "a", "o")}
    return fit(Predictor(), pred_loss(lam), sub(tr_idx), sub(va_idx), seed, lr=3e-4)


@torch.no_grad()
def predict(P, T, idx):
    return P(T["s"][idx], T["a"][idx])


def train_judge(kind, clm, T, c_tr, c_va, tr_idx, va_idx, seed):
    """kind 'heads' or 'mlp'; c_tr / c_va are the candidate vectors to judge (a, o or predicted o)."""
    bt = {"s": T["s"][tr_idx], "c": c_tr, "y": T["y"][tr_idx], "wt": T["wt"][tr_idx]}
    bv = {"s": T["s"][va_idx], "c": c_va, "y": T["y"][va_idx], "wt": T["wt"][va_idx]}
    if kind == "heads":
        m = Heads(clm)
        with torch.no_grad():
            m.b.fill_(-(m.w * m.scale * m.cos(bt["s"], bt["c"])).mean().item())
        groups = [{"params": list(m.sh.parameters()) + list(m.ah.parameters()), "lr": 2e-5},
                  {"params": [m.w, m.b], "lr": 1e-2}]
        return fit(m, judge_loss("c"), bt, bv, seed, lr=2e-5, param_groups=groups)
    return fit(MLPJudge(), judge_loss("c"), bt, bv, seed, lr=3e-4)


# ---------------------------------------------------------------- metrics

def auc(score, y):
    pos, neg = score[y == 1], score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    r = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def macro_auc(score, y, src):
    vals = [auc(score[src == s], y[src == s]) for s in np.unique(src)]
    return float(np.nanmean(vals))


def boot_auc(scores: dict, y, src, task, n=1000, seed=0):
    """Macro AUC per selector with 95% intervals, resampling tasks within each source."""
    rng = np.random.default_rng(seed)
    by_task = {}
    for i, t in enumerate(task):
        by_task.setdefault(t, []).append(i)
    tasks_by_src = {s: sorted({t for t, ss in zip(task, src) if ss == s}) for s in np.unique(src)}
    draws = {k: [] for k in scores}
    for _ in range(n):
        idx = np.concatenate([np.concatenate([by_task[t] for t in rng.choice(ts, len(ts))])
                              for ts in tasks_by_src.values()])
        for k, sc in scores.items():
            draws[k].append(macro_auc(sc[idx], y[idx], src[idx]))
    return {k: np.array(v) for k, v in draws.items()}


# ---------------------------------------------------------------- experiment

def run_split(name, rows, E, clm, holdout, seeds, split_seed=0, lams=(0.05,)):
    keep = np.array([r["label"] != 0 for r in rows])
    task = np.array([r["task"] for r in rows])
    src = np.array([r["source"] for r in rows])
    tr_t, va_t, te_t = split_tasks(rows, split_seed, holdout)
    tr = np.where(np.isin(task, list(tr_t)) & keep)[0]
    va = np.where(np.isin(task, list(va_t)) & keep)[0]
    te = np.where(np.isin(task, list(te_t)) & keep)[0]
    # predictors learn from every executed call, labelled or neutral (they need no labels)
    tr_all = np.where(np.isin(task, list(tr_t)))[0]
    va_all = np.where(np.isin(task, list(va_t)))[0]

    y = np.array([r["label"] == 1 for r in rows], dtype=np.float32)
    T = {k: torch.from_numpy(E[k]) for k in ("s", "a", "o")}
    T["y"] = torch.from_numpy(y)
    pos_rate = y[tr].mean()
    T["wt"] = torch.from_numpy(np.where(y == 1, 0.5 / pos_rate, 0.5 / (1 - pos_rate)).astype(np.float32))

    scores = {}
    with torch.no_grad():
        zs = lambda c: clm.scale * (F.normalize(clm.state_head(T["s"][te]), dim=-1)
                                    * F.normalize(clm.action_head(T[c][te]), dim=-1)).sum(-1).numpy()
        scores["clm_action"], scores["clm_true"] = zs("a"), zs("o")

    acc = {}
    cos_pred = {}
    for seed in range(seeds):
        t0 = time.time()
        out = {}
        for kind in ("heads", "mlp"):
            for c in ("a", "o"):
                m = train_judge(kind, clm, T, T[c][tr], T[c][va], tr, va, seed)
                with torch.no_grad():
                    out[f"{kind}_{'action' if c == 'a' else 'true'}"] = m(T["s"][te], T[c][te]).numpy()
        tags = [("pred", 0.0)] + [(f"pred_sigreg{lam:g}", lam) for lam in lams]
        for tag, lam in tags:
            # out-of-fold predictions for the training rows, by task
            folds = np.array_split(np.random.default_rng(seed).permutation(sorted(tr_t)), 3)
            c_tr = torch.zeros(len(tr), DIM)
            for f in folds:
                held = np.isin(task[tr], f)
                fit_idx = tr_all[~np.isin(task[tr_all], f)]
                P = train_predictor(T, fit_idx, va_all, seed, lam)
                c_tr[held] = predict(P, T, tr[held])
            P = train_predictor(T, tr_all, va_all, seed, lam)
            c_va, c_te = predict(P, T, va), predict(P, T, te)
            cos_pred.setdefault(tag, []).append(float((c_te * T["o"][te]).sum(-1).mean()))
            for kind in ("heads", "mlp"):
                m = train_judge(kind, clm, T, c_tr, c_va, tr, va, seed)
                with torch.no_grad():
                    out[f"{kind}_{tag}"] = m(T["s"][te], c_te).numpy()
        for k, v in out.items():
            acc[k] = acc.get(k, 0) + (v - v.mean()) / (v.std() + 1e-9) / seeds   # average z-scored seeds
        print(f"  {name}: seed {seed} done ({time.time() - t0:.0f}s)", flush=True)
    scores.update(acc)

    yt, st_, tt = y[te], src[te], task[te]
    draws = boot_auc(scores, yt, st_, tt)
    res = {"n_test_steps": int(len(te)), "n_test_tasks": int(len(set(tt))), "n_train_steps": int(len(tr)),
           "test_incorrect_rate": float(1 - yt.mean()), "selectors": {}, "differences": {},
           "per_source": {}, "predictor_cosine": {k: float(np.mean(v)) for k, v in cos_pred.items()},
           "cosine_action_to_true": float((T["a"][te] * T["o"][te]).sum(-1).mean())}
    for k, sc in scores.items():
        m = macro_auc(sc, yt, st_)
        lo, hi = np.nanpercentile(draws[k], [2.5, 97.5])
        res["selectors"][k] = {"auc": m, "ci": [float(lo), float(hi)]}
        res["per_source"][k] = {s: float(auc(sc[st_ == s], yt[st_ == s])) for s in np.unique(st_)}
    pairs = [("mlp_pred", "mlp_action"), ("mlp_true", "mlp_action"), ("mlp_pred", "mlp_true"),
             ("heads_pred", "heads_action"), ("heads_true", "heads_action"), ("heads_action", "mlp_action")]
    for tag, _ in tags[1:]:
        pairs += [(f"heads_{tag}", "heads_pred"), (f"mlp_{tag}", "mlp_pred"), (f"mlp_{tag}", "mlp_action")]
    for a, b in pairs:
        d = draws[a] - draws[b]
        res["differences"][f"{a} - {b}"] = {"diff": res["selectors"][a]["auc"] - res["selectors"][b]["auc"],
                                            "ci": [float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))]}
    return res


def show(name, res):
    print(f"\n{name}  ({res['n_test_tasks']} test tasks, {res['n_test_steps']} steps, "
          f"{res['test_incorrect_rate']:.0%} incorrect; {res['n_train_steps']} training steps)")
    srcs = sorted(next(iter(res["per_source"].values())))
    print(f"  {'selector':26s} {'AUC':>6s}  {'95% interval':>14s}  " + "  ".join(f"{s[:8]:>8s}" for s in srcs))
    for k, v in res["selectors"].items():
        ps = res["per_source"][k]
        print(f"  {k:26s} {v['auc']:6.3f}  [{v['ci'][0]:.3f}, {v['ci'][1]:.3f}]  "
              + "  ".join(f"{ps[s]:8.3f}" for s in srcs))
    for k, v in res["differences"].items():
        print(f"  {k}: {v['diff']:+.3f} [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}]")
    print(f"  predictor cosine to real result: " + ", ".join(f"{k} {v:.3f}" for k, v in res["predictor_cosine"].items())
          + f"   (the call's own embedding: {res['cosine_action_to_true']:.3f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", default="data/clm/CLM_v0.1-8B.pt")
    ap.add_argument("--url", default="http://127.0.0.1:8090/v1/embeddings")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--encode-only", action="store_true")
    ap.add_argument("--skip-holdout", action="store_true")
    ap.add_argument("--sigreg", default="0.05", help="comma-separated SIGReg weights, one predictor each")
    ap.add_argument("--out", default="option3.json")
    args = ap.parse_args()
    lams = tuple(float(x) for x in args.sigreg.split(",") if x)

    t0 = time.time()
    rows = load_steps()
    clm = CLM(Path(args.heads), args.url, ROOT / "clm_emb")
    E = encode(rows, clm)
    if args.encode_only:
        return
    out = {"seeds": args.seeds, "sigreg": lams}
    res = run_split("IN-DISTRIBUTION", rows, E, clm, None, args.seeds, lams=lams)
    show("IN-DISTRIBUTION", res)
    out["in_distribution"] = res
    if not args.skip_holdout:
        out["held_out_source"] = {}
        for s in SOURCES:
            r = run_split(f"HELD-OUT {s}", rows, E, clm, s, args.seeds, lams=lams)
            show(f"HELD-OUT SOURCE {s}", r)
            out["held_out_source"][s] = r
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / args.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {RESULTS / args.out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
