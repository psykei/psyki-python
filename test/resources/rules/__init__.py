import csv
from pathlib import Path
from typing import TextIO
from test.resources import rules

PATH = Path(__file__).parents[0]


def get_rule_path(filename: str) -> Path:
    return PATH / f"{filename}.txt"


def open_rule(filename: str) -> TextIO:
    return open(get_rule_path(filename))


def get_rules(name: str) -> list[str]:
    result = []
    with open(str(rules.PATH / name) + '.txt', mode="r") as file:
        reader = csv.reader(file, delimiter=';')
        for item in reader:
            result += item
    return result
