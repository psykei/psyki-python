import csv
from pathlib import Path

PATH = Path(__file__).parents[0]


def get_rules(name: str) -> list[str]:
    result = []
    with open(str(PATH / name) + '.txt', mode="r") as file:
        reader = csv.reader(file, delimiter=';')
        for item in reader:
            result += item
    return result
