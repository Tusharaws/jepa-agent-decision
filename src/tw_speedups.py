"""Caches for three pure-but-slow TextWorld internals.

TextWorld recompiles the game's PDDL on every reset and re-parses its text
templates and queries on every step. Each of these depends only on its string
input, and together they make a step cost ~100 ms in kitchens. Counterfactual
branching replays whole action prefixes, so that cost multiplies.

Set ALF_NO_SPEEDUPS=1 to run without the caches (used to verify they change nothing).
"""

import os
from copy import deepcopy
from functools import lru_cache

import fast_downward
import textworld.envs.pddl.textgen as textgen
from textworld.logic import Rule


def install():
    if os.environ.get("ALF_NO_SPEEDUPS") or getattr(install, "done", False):
        return
    install.done = True

    # 1. PDDL -> SAS translation: depends only on the game.
    pddl2sas = fast_downward.pddl2sas
    cache: dict = {}

    def cached_pddl2sas(domain, problem, verbose=False, optimize=False):
        key = (hash(domain), hash(problem), optimize)
        if key not in cache:
            if len(cache) >= 8:
                cache.clear()
            cache[key] = pddl2sas(domain, problem, verbose=verbose, optimize=optimize)
        return cache[key]

    fast_downward.pddl2sas = cached_pddl2sas

    # 2. Template parsing. The parsed symbols get a context attached later, so hand out copies.
    parse = textgen._parse_and_convert

    @lru_cache(maxsize=50_000)
    def parse_cached(text, rule_name, trace):
        return parse(text, rule_name=rule_name, trace=trace)

    def parse_and_convert(text, rule_name=None, trace=False, **kwargs):
        if kwargs:
            return parse(text, rule_name=rule_name, trace=trace, **kwargs)
        return deepcopy(parse_cached(text, rule_name, bool(trace)))

    textgen._parse_and_convert = parse_and_convert

    # 3. Query parsing. The parsed rule is only read.
    parse_query = Rule.parse_conjunctive_query
    Rule.parse_conjunctive_query = staticmethod(lru_cache(maxsize=50_000)(parse_query))
