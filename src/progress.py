"""Exact steps-to-go for ALFWorld text games, computed from the game's facts.

TextWorld's built-in planner gives the same number but takes seconds to minutes
per call in large kitchens, so it cannot be called for every candidate action.
The six ALFWorld goals have a fixed shape (fetch an object, optionally clean /
heat / cool it, put it somewhere or hold it under a lamp), so the shortest plan
can be found by enumerating the few choices involved: which object, which
target receptacle, which appliance, and where to drop anything already held.

Every action counts as one step. `solve` returns (steps, first command of one
shortest plan), or (None, None) if the goal cannot be reached.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import permutations
from typing import NamedTuple

# task type -> (property the object must end up with, capability it needs, appliance types, command verb)
TRANSFORMS = {
    "pick_clean_then_place_in_recep": ("isclean", "cleanable", ("sinkbasintype", "sinktype"), "clean"),
    "pick_heat_then_place_in_recep": ("ishot", "heatable", ("microwavetype",), "heat"),
    "pick_cool_then_place_in_recep": ("iscool", "coolable", ("fridgetype",), "cool"),
}


@dataclass(frozen=True)
class Goal:
    task_type: str
    obj_type: str            # e.g. "appletype"
    recep_type: str | None   # e.g. "countertoptype"; None for look_at_obj_in_light
    lamp_type: str | None    # e.g. "desklamptype"; only for look_at_obj_in_light

    @staticmethod
    def from_traj(task_type: str, pddl_params: dict) -> "Goal":
        t = lambda s: (s.lower() + "type") if s else None
        return Goal(task_type, t(pddl_params["object_target"]), t(pddl_params["parent_target"]),
                    t(pddl_params["toggle_target"]))


class World:
    """The facts of one game state, indexed for lookup."""

    def __init__(self, facts):
        self.agent_loc = None
        self.holding = None
        self.recep_loc: dict[str, str] = {}
        self.rtype: dict[str, str] = {}
        self.otype: dict[str, str] = {}
        self.in_recep: dict[str, set[str]] = defaultdict(set)
        self.can_contain: set[tuple[str, str]] = set()
        self.unary: dict[str, set[str]] = defaultdict(set)
        for f in facts:
            a = [v.name for v in f.arguments]
            n = f.name
            if n == "atlocation":
                self.agent_loc = a[1]
            elif n == "holds":
                self.holding = a[1]
            elif n == "receptacleatlocation":
                self.recep_loc[a[0]] = a[1]
            elif n == "receptacletype":
                self.rtype[a[0]] = a[1]
            elif n == "objecttype":
                self.otype[a[0]] = a[1]
            elif n == "inreceptacle":
                self.in_recep[a[0]].add(a[1])
            elif n == "cancontain":
                self.can_contain.add((a[0], a[1]))
            elif len(a) == 1:
                self.unary[n].add(a[0])

    def has(self, prop: str, x: str) -> bool:
        return x in self.unary[prop]

    def receps_of(self, *types: str) -> list[str]:
        return sorted(r for r, t in self.rtype.items() if t in types and r in self.recep_loc)

    def objs_of(self, otype: str) -> list[str]:
        return sorted(o for o, t in self.otype.items() if t == otype)

    def sources(self, o: str) -> list[str]:
        return sorted(r for r in self.in_recep[o] if r in self.recep_loc)


class Sim(NamedTuple):
    """A partial plan: its cost, where the agent ends up, what it opened, its first action."""
    cost: int
    loc: str | None
    opened: frozenset
    first: tuple | None

    def act(self, action: tuple) -> "Sim":
        return self._replace(cost=self.cost + 1, first=self.first or action)


class _Planner:
    def __init__(self, world: World):
        self.w = world

    def goto(self, s: Sim, r: str) -> Sim:
        loc = self.w.recep_loc[r]
        return s if s.loc == loc else s.act(("goto", r))._replace(loc=loc)

    def reach(self, s: Sim, r: str) -> Sim:
        """Stand at r with r open."""
        s = self.goto(s, r)
        if self.w.has("openable", r) and not self.w.has("opened", r) and r not in s.opened:
            s = s.act(("open", r))._replace(opened=s.opened | {r})
        return s

    def put(self, s: Sim, o: str, r: str) -> Sim:
        return self.reach(s, r).act(("put", o, r))

    def take(self, s: Sim, o: str) -> list[tuple[Sim, set[str]]]:
        """Each way of picking o up, with the receptacles the game still lists it in.

        An object can be listed in two receptacles at once (a mug on a coffee
        machine that stands on a countertop). Taking it from one leaves the
        other listing in place, and the game's win check honours it."""
        return [(self.reach(s, r).act(("take", o, r)), self.w.in_recep[o] - {r}) for r in self.w.sources(o)]

    def drop(self, s: Sim, x: str) -> list[Sim]:
        ot = self.w.otype.get(x)
        return [self.put(s, x, r) for r in sorted(self.w.recep_loc)
                if (self.w.rtype.get(r), ot) in self.w.can_contain]

    def hold(self, s: Sim, o: str, holding: str | None) -> list[tuple[Sim, set[str]]]:
        """Plans that end with o in hand, given what is in hand now.
        Each comes with the receptacles o is still listed in."""
        if holding == o:
            return [(s, set(self.w.in_recep[o]))]
        if not self.w.has("pickupable", o):
            return []
        if holding is None:
            return self.take(s, o)
        return [t for d in self.drop(s, holding) for t in self.take(d, o)]


def solve(world: World, goal: Goal) -> tuple[int | None, tuple | None]:
    w, p = world, _Planner(world)
    start = Sim(0, w.agent_loc, frozenset(), None)
    best: Sim | None = None

    def consider(s: Sim):
        nonlocal best
        if best is None or s.cost < best.cost:
            best = s

    def placeable(o: str) -> list[str]:
        return [r for r in w.receps_of(goal.recep_type) if (goal.recep_type, w.otype[o]) in w.can_contain]

    if goal.task_type == "look_at_obj_in_light":
        lamps = [(lamp, r) for lamp in w.objs_of(goal.lamp_type) if w.has("toggleable", lamp)
                 for r in w.sources(lamp)]
        for o in w.objs_of(goal.obj_type):
            for s, _ in p.hold(start, o, w.holding):
                for lamp, r in lamps:
                    t = p.goto(s, r)
                    if not w.has("istoggled", lamp):
                        t = t.act(("toggle", lamp))
                    consider(t)

    elif goal.task_type == "pick_two_obj_and_place":
        objs = w.objs_of(goal.obj_type)
        for r in w.receps_of(goal.recep_type):
            inside = [o for o in objs if r in w.in_recep[o]]
            need = 2 - len(inside)
            if need <= 0:
                return 0, None
            others = [o for o in objs if o not in inside and (goal.recep_type, w.otype[o]) in w.can_contain]
            for order in permutations(others, need):
                states, holding = [start], w.holding
                for o in order:
                    states = [p.put(t, o, r) for s in states for t, _ in p.hold(s, o, holding)]
                    holding = None
                for s in states:
                    consider(s)

    else:  # pick_and_place_simple and the clean / heat / cool variants
        prop, capability, appliance_types, verb = TRANSFORMS.get(goal.task_type, (None, None, (), None))
        appliances = w.receps_of(*appliance_types)
        for o in w.objs_of(goal.obj_type):
            if capability and not w.has(capability, o):
                continue
            ready = prop is None or w.has(prop, o)
            in_target = lambda receps: any(w.rtype.get(r) == goal.recep_type for r in receps)
            if ready and in_target(w.in_recep[o]):
                return 0, None
            for s, still_in in p.hold(start, o, w.holding):
                staged = [s] if ready else [p.goto(s, a).act((verb, o, a)) for a in appliances]
                for t in staged:
                    if in_target(still_in):
                        consider(t)  # already counted as in the target; no put needed
                    for r in placeable(o):
                        consider(p.put(t, o, r))

    return (best.cost, best.first) if best is not None else (None, None)


def render(action: tuple | None, names: dict[str, str]) -> str | None:
    """Turn a plan action into the command text the game accepts."""
    if action is None:
        return None
    verb, *args = action
    n = [names[a] for a in args]
    if verb == "goto":
        return f"go to {n[0]}"
    if verb == "open":
        return f"open {n[0]}"
    if verb == "take":
        return f"take {n[0]} from {n[1]}"
    if verb == "put":
        return f"move {n[0]} to {n[1]}"
    if verb == "toggle":
        return f"use {n[0]}"
    return f"{verb} {n[0]} with {n[1]}"  # clean / heat / cool
