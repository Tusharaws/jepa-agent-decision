"""Candidate-action proposers.

A proposer sees what an agent would see (goal, recent history, current
observation) and returns up to N candidate commands, best first. It is not
shown the game's list of admissible commands, so it can propose invalid ones.

AnthropicProposer calls the Claude API through the Anthropic SDK; it uses
whatever credentials the SDK finds (ANTHROPIC_API_KEY, or the profile from
`ant auth login`). LLMProposer talks to any OpenAI-compatible chat endpoint
(OpenAI, Ollama at http://localhost:11434/v1, and others). StubProposer needs
no model and exists only to test the pipeline.
"""

from __future__ import annotations

import json
import random
import re
import time
from dataclasses import dataclass, field

HISTORY_STEPS = 12

SYSTEM_PROMPT = """You control a household robot in a text game. Each turn you see the task, your recent actions with their results, and the latest observation.

Commands you can use (replace the parts in braces with names exactly as they appear in observations, e.g. "drawer 2", "apple 1"):
- go to {receptacle}
- open {receptacle}
- close {receptacle}
- take {object} from {receptacle}
- move {object} to {receptacle}
- clean {object} with {sinkbasin}
- heat {object} with {microwave}
- cool {object} with {fridge}
- use {lamp}
- examine {object or receptacle}
- look
- inventory

Rules of the game:
- Always write names with their number ("fridge 1", not "fridge").
- You can hold one object at a time.
- You must go to a receptacle before you can open it, take from it, or put something on it. "go to" only works for receptacles, and not for the one you are already at. To reach an object such as a lamp, go to the receptacle it is on.
- "examine {object}" only works for the object you are holding; "examine {receptacle}" only works where you are standing.
- To clean, heat or cool an object, hold it and stand at the sinkbasin, microwave or fridge.
- To look at an object under a lamp, hold the object, go to the receptacle the lamp is on, then "use {lamp}".
- If the result is "Nothing happens.", the command was invalid or not possible in the current situation. Do not send it again unchanged.

Propose exactly __N__ different candidate commands for the next step, ordered from most to least promising. Reply with JSON only, in the form {"candidates": ["...", "..."]}."""


def build_user_prompt(goal: str, history: list[tuple[str, str]], observation: str, room: str = "") -> str:
    lines = [f"Task: {goal}", ""]
    if room and history:
        lines += [f"At the start: {room}", ""]
    if history:
        lines.append("Recent steps:")
        for action, result in history[-HISTORY_STEPS:]:
            lines.append(f"> {action}\n{result}")
        lines.append("")
    lines.append(f"Latest observation: {observation}")
    return "\n".join(lines)


def parse_candidates(text: str, n: int) -> list[str]:
    """Pull the candidate list out of a model reply; tolerate code fences and stray text."""
    cands: list = []
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            cands = json.loads(m.group(0)).get("candidates", [])
        except (json.JSONDecodeError, AttributeError):
            cands = []
    if not cands:
        cands = re.findall(r'"([^"\n]{2,80})"', text.split("candidates", 1)[-1])
    out = []
    for c in cands:
        if isinstance(c, str):
            c = " ".join(c.strip().lstrip(">").split()).lower()
            if c and c not in out:
                out.append(c)
    return out[:n]


CANDIDATES_SCHEMA = {
    "type": "object",
    "properties": {"candidates": {"type": "array", "items": {"type": "string"}}},
    "required": ["candidates"],
    "additionalProperties": False,
}


@dataclass
class AnthropicProposer:
    """Claude as the proposer. Thinking is off and effort is low by default: the
    experiment wants a quick, fallible proposer so the selectors have room to help."""
    model: str = "claude-haiku-5-5"
    n: int = 5
    effort: str = "low"
    thinking: bool = False
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    _client: object = field(default=None, repr=False)

    @property
    def name(self) -> str:
        return self.model

    def propose(self, goal: str, history: list[tuple[str, str]], observation: str, room: str = "", **_) -> list[str]:
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(max_retries=5)
        for _ in range(2):
            resp = self._client.messages.create(
                model=self.model,
                max_tokens=2000 if self.thinking else 400,
                thinking={"type": "adaptive"} if self.thinking else {"type": "disabled"},
                output_config={"effort": self.effort,
                               "format": {"type": "json_schema", "schema": CANDIDATES_SCHEMA}},
                system=SYSTEM_PROMPT.replace("__N__", str(self.n)),
                messages=[{"role": "user", "content": build_user_prompt(goal, history, observation, room)}],
            )
            self.calls += 1
            u = resp.usage
            self.prompt_tokens += u.input_tokens + (u.cache_read_input_tokens or 0) + (u.cache_creation_input_tokens or 0)
            self.completion_tokens += u.output_tokens
            if resp.stop_reason in ("refusal", "max_tokens"):
                continue
            text = next((b.text for b in resp.content if b.type == "text"), "")
            cands = parse_candidates(text, self.n)
            if cands:
                return cands
        return ["look"]  # the model never produced a usable list


@dataclass
class LLMProposer:
    model: str
    n: int = 5
    temperature: float = 0.7
    base_url: str | None = None
    api_key: str | None = None
    max_retries: int = 3
    thinking: bool = True       # False sends the Qwen/vLLM switch that turns reasoning off
    min_interval: float = 0.0   # seconds between calls, to stay under a requests-per-minute limit
    _last_call: float = field(default=0.0, repr=False)
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    _client: object = field(default=None, repr=False)

    @property
    def name(self) -> str:
        return self.model

    def propose(self, goal: str, history: list[tuple[str, str]], observation: str, room: str = "", **_) -> list[str]:
        if self._client is None:
            from openai import OpenAI
            # With a custom endpoint, never fall back to OPENAI_API_KEY: that would
            # send the OpenAI key to someone else's server.
            key = self.api_key or ("none" if self.base_url else None)
            self._client = OpenAI(base_url=self.base_url, api_key=key, max_retries=8)
        extra = {} if self.thinking else {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT.replace("__N__", str(self.n))},
            {"role": "user", "content": build_user_prompt(goal, history, observation, room)},
        ]
        for _ in range(self.max_retries):
            wait = self._last_call + self.min_interval - time.time()
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.time()
            resp = self._create(messages, extra)
            self.calls += 1
            if resp.usage:
                self.prompt_tokens += resp.usage.prompt_tokens
                self.completion_tokens += resp.usage.completion_tokens
            cands = parse_candidates(resp.choices[0].message.content or "", self.n)
            if cands:
                return cands
        return ["look"]  # the model never produced a usable list

    def _create(self, messages, extra):
        """One request. The client already retries briefly; this waits out longer outages."""
        import openai
        for attempt in range(12):
            try:
                return self._client.chat.completions.create(
                    model=self.model, messages=messages, temperature=self.temperature, **extra)
            except (openai.RateLimitError, openai.InternalServerError, openai.APIConnectionError):
                if attempt == 11:
                    raise
                time.sleep(30)


@dataclass
class StubProposer:
    """Test double, not an agent: mixes the expert's action with random valid and
    invalid commands. It reads privileged game state, so data collected with it
    must not be used for the experiment."""
    n: int = 5
    seed: int = 0
    p_expert_first: float = 0.5
    name: str = "stub"

    def __post_init__(self):
        self._rng = random.Random(self.seed)

    def propose(self, goal, history, observation, *, expert_action=None, admissible=(), **_) -> list[str]:
        rng = self._rng
        pool = list(admissible) + ["open fridge 9", "take key from shelf 3"]
        cands = rng.sample(pool, min(self.n, len(pool)))
        if expert_action and expert_action not in cands:
            cands[rng.randrange(len(cands))] = expert_action
        if expert_action and rng.random() < self.p_expert_first:
            cands.remove(expert_action)
            cands.insert(0, expert_action)
        return cands
