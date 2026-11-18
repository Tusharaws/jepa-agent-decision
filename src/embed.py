"""Turn a run's step log into arrays of frozen sentence embeddings.

    uv run python -m src.embed --run NAME

Every distinct text (goal, observation, action, outcome, history line) is
embedded once with all-mpnet-base-v2 and cached in data/logs/NAME/emb/. The
dataset has one row per (step, candidate):

    state   = [goal, observation, mean of recent history lines]   3 x 768
    action  = the candidate command                               768
    outcome = what the environment returned for it                768
    label   = 0 useful, 1 error, 2 dead end

The pieces are embedded separately because the encoder truncates at 384 tokens.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .collect import LOG_ROOT, load_steps
from .env import LABELS

ENCODER = "sentence-transformers/all-mpnet-base-v2"
DIM = 768


def history_lines(history) -> list[str]:
    return [f"{a} -> {o}" for a, o in history]


def embed_texts(texts: list[str], cache_dir: Path) -> dict[str, np.ndarray]:
    """Embeddings for texts, reusing and extending the on-disk cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    idx_path, vec_path = cache_dir / "texts.json", cache_dir / "vectors.npy"
    known: list[str] = json.loads(idx_path.read_text()) if idx_path.exists() else []
    vecs = np.load(vec_path) if vec_path.exists() else np.zeros((0, DIM), dtype=np.float32)
    have = set(known)
    new = sorted(set(texts) - have)
    if new:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(ENCODER)
        print(f"embedding {len(new)} new texts ({len(known)} cached)", flush=True)
        fresh = model.encode(new, batch_size=64, normalize_embeddings=True, show_progress_bar=True,
                             convert_to_numpy=True).astype(np.float32)
        known, vecs = known + new, np.concatenate([vecs, fresh])
        idx_path.write_text(json.dumps(known))
        np.save(vec_path, vecs)
    pos = {t: i for i, t in enumerate(known)}
    return {t: vecs[pos[t]] for t in set(texts)}


@dataclass
class Dataset:
    """One row per (step, candidate). `step_id` groups the candidates of a step."""
    goal: np.ndarray        # [n, 768]
    obs: np.ndarray         # [n, 768]
    hist: np.ndarray        # [n, 768]
    action: np.ndarray      # [n, 768]
    outcome: np.ndarray     # [n, 768]
    label: np.ndarray       # [n] index into LABELS
    step_id: np.ndarray     # [n]
    rank: np.ndarray        # [n] the proposer's ranking, 0 = first choice
    game: np.ndarray        # [n] index into games
    games: list[str]
    task_type: np.ndarray   # [n] str
    verb: np.ndarray        # [n] str, first word of the action

    def __len__(self):
        return len(self.label)

    def subset(self, mask: np.ndarray) -> "Dataset":
        kw = {k: (v[mask] if isinstance(v, np.ndarray) else v) for k, v in self.__dict__.items()}
        return Dataset(**kw)


def build_dataset(run: str) -> Dataset:
    rows = load_steps(LOG_ROOT / run / "steps.jsonl")
    assert rows, f"no steps logged for run {run!r}"
    texts = set()
    for r in rows:
        texts.update([r["goal"], r["observation"], *history_lines(r["history"])])
        for c in r["candidates"]:
            texts.update([c["action"], c["outcome"]])
    emb = embed_texts(sorted(texts), LOG_ROOT / run / "emb")

    games = sorted({r["game_id"] for r in rows})
    gidx = {g: i for i, g in enumerate(games)}
    zero = np.zeros(DIM, dtype=np.float32)
    cols = {k: [] for k in ("goal", "obs", "hist", "action", "outcome", "label", "step_id", "rank", "game",
                            "task_type", "verb")}
    for sid, r in enumerate(rows):
        lines = history_lines(r["history"])
        hist = np.mean([emb[t] for t in lines], axis=0) if lines else zero
        for c in r["candidates"]:
            cols["goal"].append(emb[r["goal"]])
            cols["obs"].append(emb[r["observation"]])
            cols["hist"].append(hist)
            cols["action"].append(emb[c["action"]])
            cols["outcome"].append(emb[c["outcome"]])
            cols["label"].append(LABELS.index(c["label"]))
            cols["step_id"].append(sid)
            cols["rank"].append(c["rank"])
            cols["game"].append(gidx[r["game_id"]])
            cols["task_type"].append(r["task_type"])
            cols["verb"].append(c["action"].split()[0] if c["action"] else "")
    f32 = lambda k: np.asarray(cols[k], dtype=np.float32)
    i64 = lambda k: np.asarray(cols[k], dtype=np.int64)
    return Dataset(f32("goal"), f32("obs"), f32("hist"), f32("action"), f32("outcome"), i64("label"),
                   i64("step_id"), i64("rank"), i64("game"), games,
                   np.asarray(cols["task_type"]), np.asarray(cols["verb"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    ds = build_dataset(args.run)
    counts = np.bincount(ds.label, minlength=3)
    print(f"{len(ds)} rows, {len(np.unique(ds.step_id))} steps, {len(ds.games)} games; labels: "
          + ", ".join(f"{n} {c}" for n, c in zip(LABELS, counts)))


if __name__ == "__main__":
    main()
