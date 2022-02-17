from pathlib import Path
from typing import TextIO, Iterable

PATH = Path(__file__).parents[0]


def get_rule_path(filename: str) -> Path:
    return PATH / f"{filename}.txt"


def open_rule(filename: str) -> TextIO:
    return open(get_rule_path(filename))


def get_rules(filename: str) -> Iterable[str]:
    with open_rule(filename) as file:
        return file.readlines()
