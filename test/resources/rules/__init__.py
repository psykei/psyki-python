import csv
from pathlib import Path
from test.resources.rules.iris import PATH as IRIS_PATH
from test.resources.rules.poker import PATH as POKER_PATH
from test.resources.rules.splice_junction import PATH as SJ_PATH


PATH = Path(__file__).parents[0]

RULES_REGISTER = {
    "iris": IRIS_PATH,
    "poker": POKER_PATH,
    "splice_junction": SJ_PATH
}


def get_rules(rule_domain: str = "iris", rule_name: str = "kb") -> list[str]:
    result = []
    with open(str(RULES_REGISTER[rule_domain] / rule_name) + '.txt', mode="r") as file:
        reader = csv.reader(file, delimiter=';')
        for item in reader:
            result += item
    return result
