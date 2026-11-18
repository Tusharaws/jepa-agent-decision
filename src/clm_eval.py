"""Zero-shot CLM-v0.1-8B as a selector, on the same held-out steps as eval_offline.

    llama-server -m CLM.gguf --embedding --pooling last -c 8192 -np 4 -ngl 99 --port 8090
    uv run python -m src.clm_eval --run qwen38-main [--heads data/clm/CLM_v0.1-8B.pt]

CLM (Contrastive-LM/CLM-v0.1-8B) is a frozen Qwen3-8B encoder with last-token
pooling and two projection heads; a candidate's score is
exp(logit_scale) * cos(state_head(state), action_head(candidate)). This file
reproduces the contrastive-lm package's Engine.rank (heads.py, schema.py,
embedder.py of v0.1.0) without its vLLM dependency: the state text is
"context\\n\\nquestion", candidates are embedded verbatim, embeddings are L2
normalised before the heads, and no BOS/EOS token is added.

Selectors (no training on our data):
    clm_action         ranks the five commands
    clm_true_outcome   ranks "command -> what the game replied", using the real
                       reply (not deployable; tests CLM as a judge of outcomes,
                       the role it would play on top of a world model)

Also reported: each one combined with the proposer's ranking, with the weight
chosen on the validation games. Results go to results/RUN/clm.json; embeddings
are cached in data/logs/RUN/clm_emb/ so a re-run resumes.
"""

from __future__ import annotations

import argparse
import base64
import json
import time
from pathlib import Path

import numpy as np
import requests
import torch
import torch.nn as nn

from .collect import LOG_ROOT, load_steps
from .eval_offline import RESULTS, boot, split_games
from .embed import build_dataset

QUESTION = "Which command should the robot run next to make progress on the task?"
QUESTION_OUTCOME = "Which of these actions and results made progress on the task?"


def make_head(cfg: dict) -> nn.Module:
    """Same layout as clm.heads.make_head: inp -> (LayerNorm'd hidden blocks) -> out."""
    act = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU}[cfg.get("activation", "gelu")]
    width, depth = cfg["width"], cfg["depth"]

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Linear(cfg.get("hidden_size", 4096), width)
            self.hidden = nn.ModuleList(nn.Linear(width, width) for _ in range(depth - 2))
            self.norms = nn.ModuleList((nn.LayerNorm(width) if cfg.get("layernorm") else nn.Identity())
                                       for _ in range(depth - 2))
            self.out = nn.Linear(width, cfg.get("projection_dim", 512))
            self.act = act()

        def forward(self, x):
            x = self.act(self.inp(x))
            for lin, nrm in zip(self.hidden, self.norms):
                x = self.act(nrm(lin(x)))
            return self.out(x)

    return Head()


class CLM:
    def __init__(self, heads: Path, url: str, cache_dir: Path):
        ck = torch.load(heads, map_location="cpu", weights_only=True)
        cfg = dict(ck["cfg"], projection_dim=ck.get("projection_dim", ck["cfg"].get("projection_dim", 512)))
        self.state_head, self.action_head = make_head(cfg), make_head(cfg)
        self.state_head.load_state_dict(ck["state_head"])
        self.action_head.load_state_dict(ck["action_head"])
        self.state_head.eval(), self.action_head.eval()
        self.scale = float(torch.as_tensor(ck["logit_scale"]).float().exp().clamp(max=100.0))
        self.url, self.session = url, requests.Session()
        self.cache_dir = cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.texts_path, self.vecs_path = cache_dir / "texts.json", cache_dir / "vectors.npy"
        self.known = json.loads(self.texts_path.read_text()) if self.texts_path.exists() else []
        self.vecs = list(np.load(self.vecs_path)) if self.vecs_path.exists() else []
        self.pos = {t: i for i, t in enumerate(self.known)}

    def _fetch(self, texts: list[str]) -> list[np.ndarray]:
        r = self.session.post(self.url, json={"input": texts, "encoding_format": "float"}, timeout=600)
        r.raise_for_status()
        out = [None] * len(texts)
        for d in r.json()["data"]:
            v = np.asarray(d["embedding"], dtype=np.float32)
            out[d["index"]] = v / (np.linalg.norm(v) + 1e-12)
        return out

    def _fetch_one(self, text: str) -> np.ndarray:
        """One text; if the server rejects it as too long, keep its last 60% and retry
        (CLM's training kept the last tokens). The cache key stays the full text."""
        cut = text
        for _ in range(6):
            try:
                return self._fetch([cut])[0]
            except requests.HTTPError:
                cut = cut[-int(len(cut) * 0.6):]
        raise RuntimeError(f"could not embed text of {len(text)} chars")

    def save(self):
        self.texts_path.write_text(json.dumps(self.known))
        np.save(self.vecs_path, np.stack(self.vecs))

    def embed(self, texts: list[str], batch: int = 8, log_every: int = 200) -> np.ndarray:
        todo = [t for t in dict.fromkeys(texts) if t not in self.pos]
        t0 = time.time()
        for i in range(0, len(todo), batch):
            chunk = todo[i:i + batch]
            try:
                got = self._fetch(chunk)
            except requests.RequestException:
                got = [self._fetch_one(t) for t in chunk]
            for t, v in zip(chunk, got):
                self.pos[t] = len(self.known)
                self.known.append(t)
                self.vecs.append(v)
            done = i + len(chunk)
            if done % log_every < batch or done == len(todo):
                self.save()
                print(f"  embedded {done}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)
        return np.stack([self.vecs[self.pos[t]] for t in texts])

    @torch.no_grad()
    def scores(self, states: np.ndarray, cands: np.ndarray) -> np.ndarray:
        """Row-wise scale * cos(state_head(s_i), action_head(c_i))."""
        f = lambda h, x: nn.functional.normalize(h(torch.from_numpy(x)), dim=-1)
        return (self.scale * (f(self.state_head, states) * f(self.action_head, cands)).sum(-1)).numpy()


def state_context(r: dict) -> str:
    """Same information the proposer saw: task, starting room, recent history, latest observation."""
    lines = [f"Task: {r['goal']}"]
    if r["history"]:
        lines.append(f"At the start: {r['room']}")
        lines.append("Recent actions and results:\n" + "\n".join(f"> {a}\n{o}" for a, o in r["history"]))
    lines.append(f"Latest observation: {r['observation']}")
    return "\n\n".join(lines)


def pick_useful(score: np.ndarray, rank: np.ndarray, useful: np.ndarray, step: np.ndarray) -> np.ndarray:
    """Per step: is the top-scored candidate useful. Ties go to the proposer's higher-ranked candidate."""
    order = np.lexsort((rank, -score, step))
    first = np.r_[True, step[order][1:] != step[order][:-1]]
    return useful[order][first].astype(float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--heads", default="data/clm/CLM_v0.1-8B.pt")
    ap.add_argument("--url", default="http://127.0.0.1:8090/v1/embeddings")
    ap.add_argument("--split-seed", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    rows = load_steps(LOG_ROOT / args.run / "steps.jsonl")
    ds = build_dataset(args.run)  # for the identical game split
    train_g, val_g, test_g = split_games(ds, args.split_seed)
    names = ds.games
    split_of = {**{names[g]: "val" for g in val_g}, **{names[g]: "test" for g in test_g}}
    clm = CLM(Path(args.heads), args.url, LOG_ROOT / args.run / "clm_emb")

    # sanity check on the model card's example
    s = clm.embed([f"What causes tides on Earth?"])
    c = clm.embed(["The Moon's gravitational pull.", "Photosynthesis in plants.", "Because the Earth is round."])
    print("model-card example scores (should rank the Moon first):",
          np.round(clm.scores(np.repeat(s, 3, 0), c), 1), flush=True)

    out = {}
    for part in ("val", "test"):
        steps = [r for r in rows if split_of.get(r["game_id"]) == part]
        cols = {k: [] for k in ("step", "game", "rank", "useful", "s_act", "s_out")}
        print(f"{part}: {len(steps)} steps", flush=True)
        ctx = [state_context(r) for r in steps]
        st_act = clm.embed([f"{c}\n\n{QUESTION}" for c in ctx])
        st_out = clm.embed([f"{c}\n\n{QUESTION_OUTCOME}" for c in ctx])
        acts = [c["action"] for r in steps for c in r["candidates"]]
        outs = [f"{c['action']} -> {c['outcome']}" for r in steps for c in r["candidates"]]
        e_act, e_out = clm.embed(acts), clm.embed(outs)
        k = 0
        for i, r in enumerate(steps):
            for c in r["candidates"]:
                cols["step"].append(i)
                cols["game"].append(r["game_id"])
                cols["rank"].append(c["rank"])
                cols["useful"].append(c["label"] == "useful")
                k += 1
        idx = np.repeat(np.arange(len(steps)), [len(r["candidates"]) for r in steps])
        cols["s_act"] = clm.scores(st_act[idx], e_act)
        cols["s_out"] = clm.scores(st_out[idx], e_out)
        out[part] = {k: np.asarray(v) for k, v in cols.items()}
        clm.save()

    v, t = out["val"], out["test"]
    games = np.array([g for g, first in zip(t["game"], np.r_[True, t["step"][1:] != t["step"][:-1]]) if first])
    res = {"run": args.run, "n_test_steps": int(t["step"].max() + 1), "n_test_games": len(set(games)),
           "selectors": {}, "weights": {}}
    sel = {"llm_first": pick_useful(-t["rank"].astype(float), t["rank"], t["useful"], t["step"]),
           "oracle": pick_useful(t["useful"].astype(float), t["rank"], t["useful"], t["step"])}
    for name, key in (("clm_action", "s_act"), ("clm_true_outcome", "s_out")):
        sel[name] = pick_useful(t[key], t["rank"], t["useful"], t["step"])
        # combine with the proposer's ranking: score - w * rank, w chosen on validation games
        ws = [0, 0.5, 1, 2, 4, 8, 1e9]
        val_use = [pick_useful(v[key] - w * v["rank"], v["rank"], v["useful"], v["step"]).mean() for w in ws]
        w = ws[int(np.argmax(val_use))]
        res["weights"][name + "_plus_rank"] = w
        sel[name + "_plus_rank"] = pick_useful(t[key] - w * t["rank"], t["rank"], t["useful"], t["step"])

    print(f"\nCLM ZERO-SHOT, in-distribution test split ({res['n_test_games']} games, {res['n_test_steps']} steps)")
    print(f"  {'selector':26s} {'useful':>7s}  {'95% interval':>14s}  {'vs first choice':>22s}")
    for name, u in sel.items():
        m, lo, hi = boot(u, games)
        d, dlo, dhi = boot(u - sel["llm_first"], games)
        res["selectors"][name] = {"useful": m, "useful_ci": [lo, hi], "vs_llm": d, "vs_llm_ci": [dlo, dhi]}
        print(f"  {name:26s} {m:7.1%}  [{lo:5.1%}, {hi:5.1%}]  {d:+6.1%} [{dlo:+5.1%}, {dhi:+5.1%}]")
    print("  rank weights chosen on validation:", res["weights"])
    path = RESULTS / args.run / "clm.json"
    path.write_text(json.dumps(res, indent=2))
    print(f"\nwrote {path}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
