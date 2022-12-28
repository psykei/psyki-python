import re
from test.resources.data.splice_junction import AGGREGATE_FEATURE_MAPPING

VARIABLE_BASE_NAME = 'X'
AND_SYMBOL = ' , '
OR_SYMBOL = ' ; '
NOT_SYMBOL = '¬'
LESS_EQUAL_SYMBOL = ' =< '
PLUS_SYMBOL = ' + '
STATIC_IMPLICATION_SYMBOL = ' <- '
MUTABLE_IMPLICATION_SYMBOL = ' <-- '
STATIC_RULE_SYMBOL = '::-'
MUTABLE_RULE_SYMBOL = ':-'
RULE_DEFINITION_SYMBOLS = (STATIC_RULE_SYMBOL, MUTABLE_RULE_SYMBOL)
RULE_DEFINITION_SYMBOLS_REGEX = '(' + '|'.join(RULE_DEFINITION_SYMBOLS) + ')'


def next_index(index: str, indices: list[int], offset: int) -> int:
    new_index: int = int(index) + offset
    modified: bool = False
    while new_index not in indices:
        new_index += 1
        modified = True
    return new_index + previous_holes(indices, indices.index(new_index)) if not modified else new_index


def previous_holes(l: list[int], i: int) -> int:
    j = 0
    for k in list(range(0, i)):
        if l[k] + 1 != l[k + 1]:
            j += 1
    return j


def explicit_variables(e: str) -> str:
    result = ''
    for key in AGGREGATE_FEATURE_MAPPING.keys():
        if key.lower() in e:
            values = [v for v in AGGREGATE_FEATURE_MAPPING[key]]
            if len(values) > 1:
                result += AND_SYMBOL.join(
                    NOT_SYMBOL + '(' + re.sub(key.lower(), value.lower(), e) + ')' for value in values)
    return NOT_SYMBOL + '(' + result + ')' if result != '' else e


def replace(s: str, e: str) -> str:
    values = [v for v in AGGREGATE_FEATURE_MAPPING[s]]
    return '(' + OR_SYMBOL.join('(' + re.sub(s.lower(), value.lower(), e) + ')' for value in values) + ')'
