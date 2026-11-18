"""The learned selectors, all on frozen 768-d sentence embeddings.

JEPA arm
    Predictor   (state, action) -> predicted outcome embedding      cosine loss
    ValueHead   (outcome embedding, goal) -> useful / error / dead end

Decision arm
    DecisionModel   (state, action) -> useful / error / dead end
    with aux=True it also predicts the outcome embedding from its trunk, as an
    auxiliary loss only (the ablation that separates "extra training signal"
    from "deciding through the predicted outcome").

state = [goal, observation, mean of recent history].

Rank-aware variants (extra=N_RANKS) also receive the proposer's ranking of the
candidate as a one-hot input at the decision layer: the value head in the JEPA
arm, the trunk in the decision arm. They re-rank the proposer's list instead of
replacing its judgment. The predictor never sees the rank.
"""

from __future__ import annotations

import copy
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DIM = 768
N_RANKS = 5
DEVICE = os.environ.get("ALF_DEVICE") or ("mps" if torch.backends.mps.is_available() else "cpu")


def mlp(sizes: list[int], dropout: float = 0.1) -> nn.Sequential:
    layers = []
    for i in range(len(sizes) - 2):
        layers += [nn.Linear(sizes[i], sizes[i + 1]), nn.GELU(), nn.Dropout(dropout)]
    layers.append(nn.Linear(sizes[-2], sizes[-1]))
    return nn.Sequential(*layers)


class Predictor(nn.Module):
    def __init__(self, hidden: int = 1024):
        super().__init__()
        self.net = mlp([4 * DIM, hidden, hidden, DIM])

    def forward(self, state, action):
        return F.normalize(self.net(torch.cat([state, action], -1)), dim=-1)


class ValueHead(nn.Module):
    def __init__(self, hidden: int = 512, extra: int = 0):
        super().__init__()
        self.extra = extra
        self.net = mlp([2 * DIM + extra, hidden, 3])

    def forward(self, outcome, goal, rank=None):
        parts = [outcome, goal] + ([rank] if self.extra else [])
        return self.net(torch.cat(parts, -1))


class DecisionModel(nn.Module):
    def __init__(self, hidden: int = 1024, aux: bool = False, extra: int = 0):
        super().__init__()
        self.extra = extra
        self.trunk = nn.Sequential(mlp([4 * DIM + extra, hidden, hidden, hidden]), nn.GELU())
        self.head = nn.Linear(hidden, 3)
        self.aux = nn.Linear(hidden, DIM) if aux else None

    def forward(self, state, action, rank=None):
        parts = [state, action] + ([rank] if self.extra else [])
        h = self.trunk(torch.cat(parts, -1))
        return self.head(h), (F.normalize(self.aux(h), dim=-1) if self.aux is not None else None)


def n_params(*modules) -> int:
    return sum(p.numel() for m in modules for p in m.parameters())


def tensors(ds) -> dict[str, torch.Tensor]:
    t = lambda a: torch.from_numpy(a).to(DEVICE)
    return {"state": t(np.concatenate([ds.goal, ds.obs, ds.hist], 1)), "goal": t(ds.goal),
            "action": t(ds.action), "outcome": t(ds.outcome), "label": t(ds.label),
            "rank": t(np.eye(N_RANKS, dtype=np.float32)[np.clip(ds.rank, 0, N_RANKS - 1)])}


def fit(model: nn.Module, loss_fn, train: dict, val: dict, seed: int, epochs: int = 80, patience: int = 8,
        batch: int = 256, lr: float = 3e-4) -> nn.Module:
    """Train with AdamW and keep the weights from the epoch with the best validation loss."""
    torch.manual_seed(seed)
    model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    n = len(train["label"])
    gen = torch.Generator().manual_seed(seed)
    best, best_state, bad = float("inf"), None, 0
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=gen).to(DEVICE)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            loss = loss_fn(model, {k: v[idx] for k, v in train.items()})
            opt.zero_grad()
            loss.backward()
            opt.step()
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


def predictor_loss(model, b):
    return (1 - (model(b["state"], b["action"]) * b["outcome"]).sum(-1)).mean()


def value_loss(model, b):
    return F.cross_entropy(model(b["outcome"], b["goal"], b["rank"]), b["label"])


def decision_loss(model, b):
    logits, aux = model(b["state"], b["action"], b["rank"])
    loss = F.cross_entropy(logits, b["label"])
    if aux is not None:
        loss = loss + (1 - (aux * b["outcome"]).sum(-1)).mean()
    return loss


@torch.no_grad()
def predict_outcomes(predictor, t: dict) -> torch.Tensor:
    return predictor(t["state"], t["action"])


def train_all(train_ds, val_ds, seed: int) -> dict:
    """Train every model for one seed. Returns the fitted modules."""
    tr, va = tensors(train_ds), tensors(val_ds)
    predictor = fit(Predictor(), predictor_loss, tr, va, seed)
    value_true = fit(ValueHead(), value_loss, tr, va, seed)
    # A second value head trained on the predictor's own outputs, which is what it sees at run time.
    tr_p = dict(tr, outcome=predict_outcomes(predictor, tr))
    va_p = dict(va, outcome=predict_outcomes(predictor, va))
    value_pred = fit(ValueHead(), value_loss, tr_p, va_p, seed)
    decision = fit(DecisionModel(), decision_loss, tr, va, seed)
    decision_aux = fit(DecisionModel(aux=True), decision_loss, tr, va, seed)
    value_pred_rank = fit(ValueHead(extra=N_RANKS), value_loss, tr_p, va_p, seed)
    decision_rank = fit(DecisionModel(extra=N_RANKS), decision_loss, tr, va, seed)
    return {"predictor": predictor, "value_true": value_true, "value_pred": value_pred,
            "decision": decision, "decision_aux": decision_aux,
            "value_pred_rank": value_pred_rank, "decision_rank": decision_rank}


@torch.no_grad()
def score_all(models: dict, ds) -> dict[str, np.ndarray]:
    """Class probabilities [n, 3] for each selector on a dataset, plus predictor cosine."""
    t = tensors(ds)
    pred = predict_outcomes(models["predictor"], t)
    p = lambda logits: F.softmax(logits, -1).cpu().numpy()
    return {
        "jepa_rank": p(models["value_pred_rank"](pred, t["goal"], t["rank"])),
        "decision_rank": p(models["decision_rank"](t["state"], t["action"], t["rank"])[0]),
        "jepa": p(models["value_pred"](pred, t["goal"])),               # value head trained on predictions
        "jepa_true_trained": p(models["value_true"](pred, t["goal"])),  # trained on true, run on predictions
        "value_on_true_outcome": p(models["value_true"](t["outcome"], t["goal"])),  # upper bound, not deployable
        "decision": p(models["decision"](t["state"], t["action"])[0]),
        "decision_aux": p(models["decision_aux"](t["state"], t["action"])[0]),
        "_pred_cos": (pred * t["outcome"]).sum(-1).cpu().numpy(),
    }
