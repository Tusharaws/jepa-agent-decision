"""ALFWorld text-environment wrapper.

Gives each step an outcome label derived from the exact steps-to-go (see
progress.py), and supports counterfactual "what would this other action have
done" queries by resetting a second copy of the game and replaying the action
prefix.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import textworld
from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos

from .progress import Goal, World, render, solve
from .tw_speedups import install as install_speedups

install_speedups()

DATA_ROOT = Path(__file__).resolve().parent.parent / "data" / "alfworld" / "json_2.1.1"
SPLITS = ("train", "valid_seen", "valid_unseen")
TASK_TYPES = (
    "pick_and_place_simple",
    "look_at_obj_in_light",
    "pick_clean_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_two_obj_and_place",
)

USEFUL, ERROR, DEAD_END = "useful", "error", "dead_end"
LABELS = (USEFUL, ERROR, DEAD_END)
ERROR_FEEDBACK = "Nothing happens."

_TASK_RE = re.compile(r"Your task is to: (.+?)\s*$", re.S)


@dataclass(frozen=True)
class Game:
    path: str
    split: str
    task_type: str

    @property
    def game_id(self) -> str:
        p = Path(self.path)
        return f"{self.split}/{p.parent.parent.name}/{p.parent.name}"

    @cached_property
    def goal_spec(self) -> Goal:
        with open(Path(self.path).with_name("traj_data.json")) as f:
            return Goal.from_traj(self.task_type, json.load(f)["pddl_params"])


def list_games(split: str) -> list[Game]:
    """Solvable games of a split, in a stable order."""
    assert split in SPLITS, split
    games = []
    for gf in sorted((DATA_ROOT / split).glob("*/*/game.tw-pddl")):
        task_dir = gf.parent.parent.name
        if "movable" in task_dir or "Sliced" in task_dir:
            continue  # not supported by the text engine (same filter as alfworld)
        task_type = task_dir.split("-")[0]
        if task_type not in TASK_TYPES:
            continue
        with open(gf) as f:
            if not json.load(f).get("solvable", False):
                continue
        games.append(Game(str(gf), split, task_type))
    return games


def label_outcome(feedback: str, steps_before: int | None, steps_after: int | None, won: bool) -> str:
    """useful: fewer steps remain than before (or the task is won).
    error: the environment rejected the command.
    dead_end: anything else, including actions that undo progress."""
    if won:
        return USEFUL
    if feedback.strip() == ERROR_FEEDBACK:
        return ERROR
    if steps_before is not None and steps_after is not None and steps_after < steps_before:
        return USEFUL
    return DEAD_END


@dataclass
class Outcome:
    action: str
    feedback: str
    steps_before: int | None
    steps_after: int | None
    won: bool
    label: str


@dataclass
class Observation:
    feedback: str
    steps_to_go: int | None        # length of the shortest plan from here
    next_expert_action: str | None  # first command of one shortest plan
    admissible: list[str]
    won: bool


def _start(gamefile: str, planner: bool = False):
    infos = textworld.EnvInfos(won=True, admissible_commands=True, policy_commands=planner)
    return textworld.start(gamefile, infos, wrappers=[AlfredDemangler(shuffle=False), AlfredInfos])


@dataclass
class AlfEnv:
    """One game. `step` advances it; `branch` evaluates actions without advancing it."""

    game: Game
    goal: str = ""
    actions: list[str] = field(default_factory=list)
    obs: Observation | None = None

    def __post_init__(self):
        self._env = _start(self.game.path)
        self._branch_env = None  # created on first use

    def _progress(self, state) -> tuple[int | None, str | None]:
        if state["won"]:
            return 0, None
        steps, first = solve(World(state["_facts"]), self.game.goal_spec)
        names = {k: info.name for k, info in state["_entity_infos"].items()}
        return steps, render(first, names)

    def _observe(self, state, feedback: str | None = None) -> Observation:
        steps, nxt = self._progress(state)
        return Observation((feedback if feedback is not None else state["feedback"]).strip(), steps, nxt,
                           list(state["admissible_commands"]), bool(state["won"]))

    def reset(self) -> Observation:
        state = self._env.reset()
        text = state["feedback"]
        m = _TASK_RE.search(text)
        self.goal = m.group(1).strip() if m else ""
        # Drop the banner and the task line; keep the room description.
        room = text.split("=-", 2)[-1].split("Your task is to:")[0].strip()
        self.actions = []
        self.obs = self._observe(state, room)
        return self.obs

    def step(self, action: str) -> Outcome:
        before = self.obs.steps_to_go
        state, _, _ = self._env.step(action)
        self.actions.append(action)
        self.obs = self._observe(state)
        return Outcome(action, self.obs.feedback, before, self.obs.steps_to_go, self.obs.won,
                       label_outcome(self.obs.feedback, before, self.obs.steps_to_go, self.obs.won))

    def branch(self, candidates: list[str]) -> list[Outcome]:
        """Outcome of each candidate from the current state, leaving this env untouched.

        Replays the action prefix on a second env. A rejected command leaves the
        state unchanged, so the replay is reused for the next candidate."""
        if self._branch_env is None:
            self._branch_env = _start(self.game.path)
        env = self._branch_env
        before = self.obs.steps_to_go
        out, at_prefix = [], False
        for cand in candidates:
            if not at_prefix:
                env.reset()
                for a in self.actions:
                    env.step(a)
            state, _, _ = env.step(cand)
            fb = state["feedback"].strip()
            won = bool(state["won"])
            if fb == ERROR_FEEDBACK:
                after = before
            else:
                after, _ = self._progress(state)
            out.append(Outcome(cand, fb, before, after, won, label_outcome(fb, before, after, won)))
            at_prefix = fb == ERROR_FEEDBACK
        return out

    def close(self):
        for e in (self._env, self._branch_env):
            if e is not None:
                e.close()
