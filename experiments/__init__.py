"""Each module exposes ``cases(small=False) -> list[Case]``."""

import importlib

from lab.case import Case

NAMES = ["regex", "nbody", "groupby", "montecarlo", "regex_dfa", "graphs", "levenshtein", "vm"]


def load(name: str, small: bool = False) -> list[Case]:
    return importlib.import_module(f"experiments.{name}").cases(small=small)


def all_cases(small: bool = False) -> list[Case]:
    return [case for name in NAMES for case in load(name, small)]
