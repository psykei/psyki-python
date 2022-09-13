from pathlib import Path
from typing import Iterable, Callable
import numpy as np
from test.resources.data.splice_junction import get_indices
from test.resources.rules.utils import *

PATH = Path(__file__).parents[0]
INDEX_IDENTIFIER = '@'
NOT_IDENTIFIER = 'not'
CONFUSION_MATRIX = np.array([[295, 0, 473], [25, 31, 711], [3, 0, 1652]])
CLASS_LABELS = ['IE', 'EI', 'N']


def parse_clause(rest: str, rhs: str = '', aggregation: str = AND_SYMBOL) -> str:
    for j, clause in enumerate(rest.split(',')):
        index = re.match(INDEX_IDENTIFIER + '[-]?[0-9]*', clause)
        negation = re.match(NOT_IDENTIFIER, clause)
        n = re.match('[0-9]*of', clause)
        if index is not None:
            index = clause[index.regs[0][0]:index.regs[0][1]]
            clause = clause[len(index):]
            clause = re.sub('\'', '', clause)
            index = index[1:]
            rhs += aggregation.join(explicit_variables(
                VARIABLE_BASE_NAME + ('_' if next_index(index, get_indices(), i) < 0 else '') +
                str(abs(next_index(index, get_indices(), i))) +
                ' = ' + value.lower()) for i, value in enumerate(clause))
        elif negation is not None:
            new_clause = re.sub(NOT_IDENTIFIER, NOT_SYMBOL, clause)
            new_clause = re.sub('-', '_', new_clause.lower())
            new_clause = re.sub('\)', '())', new_clause)
            rhs += new_clause
        elif n is not None:
            new_clause = clause[n.regs[0][1]:]
            new_clause = re.sub('\(|\)', '', new_clause)
            inner_clause = parse_clause(new_clause, rhs, PLUS_SYMBOL)
            inner_clause = '(' + ('), (').join(e for e in inner_clause.split(PLUS_SYMBOL)) + ')'
            n = clause[n.regs[0][0]:n.regs[0][1] - 2]
            rhs += 'm_of_n(' + n + ', ' + inner_clause + ')'
        else:
            rhs += re.sub('-', '_', clause.lower()) + '()'
        if j < len(rest.split(',')) - 1:
            rhs += AND_SYMBOL
    return rhs


def get_binary_datalog_rules(rules: Iterable[str]) -> Iterable[str]:
    results = []
    term_regex = '[a-z]+'
    variable_regex = VARIABLE_BASE_NAME + '[_]?[0-9]+'
    regex = variable_regex + '[ ]?=[ ]?' + term_regex
    for rule in rules:
        tmp_rule = rule
        partial_result = ''
        while re.search(regex, tmp_rule) is not None:
            match = re.search(regex, tmp_rule)
            start, end = match.regs[0]
            matched_string = tmp_rule[start:end]
            ante = tmp_rule[:start]
            medio = matched_string[:re.search(variable_regex, matched_string).regs[0][1]] + \
                    matched_string[re.search(term_regex, matched_string).regs[0][0]:] + ' = 1'
            partial_result += ante + medio
            tmp_rule = tmp_rule[end:]
        partial_result += tmp_rule
        results.append(partial_result)
    return results


def get_splice_junction_datalog_rules(rules: Iterable[str]) -> Iterable[str]:
    return get_datalog_rules(rules, {'ei', 'ie', 'n'}, parse_clause)


def get_datalog_rules(rules: Iterable[str], class_labels: set[str], parse_clause_f: Callable) -> Iterable[str]:
    results = []
    for rule in rules:
        rule = re.sub(r' |\.', '', rule)
        name, op, rest = re.split(RULE_DEFINITION_SYMBOLS_REGEX, rule)
        name = re.sub('-', '_', name.lower())
        rhs = parse_clause_f(rest)
        if name in class_labels:
            results.append('class(' + name + ')' +
                           (STATIC_IMPLICATION_SYMBOL if op == STATIC_RULE_SYMBOL else MUTABLE_IMPLICATION_SYMBOL)
                           + rhs)
        results.append(name + '(' + ')' +
                       (STATIC_IMPLICATION_SYMBOL if op == STATIC_RULE_SYMBOL else MUTABLE_IMPLICATION_SYMBOL) + rhs)
    return results

