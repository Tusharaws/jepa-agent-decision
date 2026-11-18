"""AgentProcessBench -> one row per labelled tool-call step.

    uv run python -m src.apb            # writes data/agentprocessbench/steps.jsonl and prints stats

AgentProcessBench (Fan et al. 2026, arXiv 2603.14465; MIT licence) has 1,000
agent runs, 250 from each of bfcl, gaia_dev, hotpotqa and tau2: 50 tasks per
source, each attempted by 5 different LLM agents. Every assistant message has a
human label: +1 correct, 0 neutral, -1 incorrect. Once a step is labelled -1,
later steps that follow from that mistake also get -1 until the agent recovers.

Only assistant steps that call a tool are kept, because only they have a
result to predict. For each one:

    state     what the agent had seen before the step: the first user message,
              then the most recent conversation, cut to the last STATE_CHARS
    action    the tool call(s), "name(arguments)"
    outcome   the tool response(s), cut to OUTCOME_CHARS
    label     +1 / 0 / -1

The record's `question` field is not used: for tau2 it holds the simulated
user's hidden instructions, which the agent never saw.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "data" / "agentprocessbench"
SOURCES = ("bfcl", "gaia_dev", "hotpotqa", "tau2")
STATE_CHARS = 5000      # about 1,300 tokens
TASK_CHARS = 1500
OUTCOME_CHARS = 1500
LINE_CHARS = 600        # per message inside the history


def call_text(calls: list[dict]) -> str:
    out = []
    for c in calls:
        f = c.get("function", c)
        args = f.get("arguments", "")
        if not isinstance(args, str):
            args = json.dumps(args, ensure_ascii=False)
        out.append(f"{f.get('name', '?')}({args})")
    return "; ".join(out)


def clip(text: str, n: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[:n] + " …"


def line(m: dict) -> str | None:
    role = m["role"]
    if role == "system":
        return None
    if role == "user":
        return f"User: {clip(m.get('content') or '', LINE_CHARS)}"
    if role == "tool":
        return f"Result of {m.get('name') or 'tool'}: {clip(m.get('content') or '', LINE_CHARS)}"
    parts = []
    if m.get("content"):
        parts.append(clip(m["content"], LINE_CHARS))
    if m.get("tool_calls"):
        parts.append("calls " + clip(call_text(m["tool_calls"]), LINE_CHARS))
    return "Assistant: " + " ".join(parts) if parts else None


def state_text(msgs: list[dict], i: int) -> str:
    first_user = next((m.get("content") or "" for m in msgs[:i] if m["role"] == "user"), "")
    hist = [l for l in (line(m) for m in msgs[:i]) if l]
    body = "\n".join(hist)
    if len(body) > STATE_CHARS:
        body = "…\n" + body[-STATE_CHARS:].split("\n", 1)[-1]
    return f"Task: {clip(first_user, TASK_CHARS)}\n\nConversation so far:\n{body}"


def steps_of(run: dict, source: str) -> list[dict]:
    msgs, labels = run["messages"], run["step_labels"]
    out = []
    for k, label in sorted(labels.items(), key=lambda kv: int(kv[0])):
        i = int(k)
        m = msgs[i]
        if m["role"] != "assistant" or not m.get("tool_calls"):
            continue
        ids = {c.get("id") for c in m["tool_calls"]}
        replies, j = [], i + 1
        while j < len(msgs) and msgs[j]["role"] == "tool":
            if not ids - {None} or msgs[j].get("tool_call_id") in ids:
                replies.append(str(msgs[j].get("content") or ""))
            j += 1
        if not replies:
            continue
        out.append({
            "source": source,
            "run": f"{source}/{run['total_index']}",
            "task": f"{source}/{run['query_index']}",
            "agent": run["sample_index"],
            "msg_index": i,
            "first_call": not any(x["role"] == "assistant" and x.get("tool_calls") for x in msgs[:i]),
            "state": state_text(msgs, i),
            "action": call_text(m["tool_calls"]),
            "outcome": clip(" | ".join(replies), OUTCOME_CHARS),
            "label": int(label),
            "run_label": run["final_label"],
        })
    return out


def load_steps() -> list[dict]:
    path = ROOT / "steps.jsonl"
    if not path.exists():
        build()
    return [json.loads(l) for l in open(path)]


def build() -> list[dict]:
    rows = []
    for s in SOURCES:
        with open(ROOT / f"{s}.jsonl") as f:
            for l in f:
                rows.extend(steps_of(json.loads(l), s))
    with open(ROOT / "steps.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return rows


def main():
    rows = build()
    print(f"{len(rows)} tool-call steps written to {ROOT / 'steps.jsonl'}")
    for s in SOURCES:
        rs = [r for r in rows if r["source"] == s]
        firsts = [r for r in rs if r["first_call"]]
        print(f"{s:9s} {len(rs):5d} steps, labels {dict(sorted(Counter(r['label'] for r in rs).items()))}, "
              f"{len({r['task'] for r in rs})} tasks; first calls {len(firsts)} "
              f"(labels {dict(sorted(Counter(r['label'] for r in firsts).items()))})")
    n = lambda key: sum(len(r[key]) for r in rows) / len(rows)
    print(f"mean chars: state {n('state'):.0f}, action {n('action'):.0f}, outcome {n('outcome'):.0f}")


if __name__ == "__main__":
    main()
