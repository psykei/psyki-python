from __future__ import annotations
from abc import ABC
from typing import Iterable
from psyki.ski import Formula
import psyki.logic.datalog as datalog


# TODO: refactoring
def optimize_datalog_formula(formula: Formula):
    if isinstance(formula, datalog.grammar.Expression):
        lhs = formula.lhs
        rhs = formula.rhs
        op = formula.op
        if op in ('∧', '∨', '+') and len(formula.nary) == 0:
            if isinstance(lhs, datalog.grammar.Expression):
                if lhs.op == op:
                    optimize_datalog_formula(lhs)
                    for clause in lhs.nary:
                        formula.nary.append(clause)
                    formula.lhs = None
                else:
                    formula.nary.append(lhs)
            else:
                formula.nary.append(lhs)
            if isinstance(rhs, datalog.grammar.Expression):
                if rhs.op == op:
                    optimize_datalog_formula(rhs)
                    for clause in rhs.nary:
                        formula.nary.append(clause)
                    formula.rhs = None
                else:
                    formula.nary.append(rhs)
            else:
                formula.nary.append(rhs)
        else:
            optimize_datalog_formula(lhs)
            optimize_datalog_formula(rhs)
    else:
        if hasattr(formula, 'lhs'):
            optimize_datalog_formula(formula.lhs)
        if hasattr(formula, 'rhs'):
            optimize_datalog_formula(formula.rhs)
        if hasattr(formula, 'predicate'):
            optimize_datalog_formula(formula.predicate)


class DatalogFormula(Formula):

    def __init__(self, lhs: DefinitionClause, rhs: Clause, op: str = '←'):
        self.lhs: DefinitionClause = lhs
        self.rhs: Clause = rhs
        self.op: str = op

    def __str__(self) -> str:
        return str(self.lhs) + self.op + str(self.rhs)

    def copy(self) -> Formula:
        return DatalogFormula(self.lhs.copy(), self.rhs.copy(), self.op)


class DefinitionClause(Formula):

    def __init__(self, predication: str, arg: Argument):
        self.predication: str = predication
        self.arg: Argument = arg

    def __str__(self) -> str:
        return self.predication + '(' + str(self.arg) + ')'

    def copy(self) -> Formula:
        return DefinitionClause(self.predication, self.arg)


class Clause(Formula, ABC):
    pass


class Expression(Clause):

    def __init__(self, lhs: Clause, rhs: Clause, op: str, nary: Iterable[Clause] = []):
        self.lhs: Clause = lhs
        self.rhs: Clause = rhs
        self.nary: list[Clause] = list(nary)
        self.op: str = op

    def __str__(self) -> str:
        if len(self.nary) == 0:
            return '((' + str(self.lhs) + ')' + self.op + '(' + str(self.rhs) + '))'
        else:
            return '(' + self.op + '(' + ','.join(str(clause) for clause in self.nary) + ')'

    def copy(self) -> Formula:
        return Expression(self.lhs.copy(), self.rhs.copy(), self.op, [c.copy() for c in self.nary])


class Literal(Clause, ABC):
    pass


class Negation(Literal):

    def __init__(self, predicate: Clause):
        self.predicate: Clause = predicate

    def __str__(self) -> str:
        return 'neg(' + str(self.predicate) + ')'

    def copy(self) -> Formula:
        return Negation(self.predicate.copy())


class Predicate(Clause, ABC):
    pass


class Unary(Predicate):

    def __init__(self, name: str):
        self.name: str = name

    def __str__(self) -> str:
        return self.name

    def copy(self) -> Formula:
        return Unary(self.name)


class Nary(Predicate):

    def __init__(self, name: str, arg: Argument):
        self.name: str = name
        self.arg: Argument = arg

    def __str__(self) -> str:
        return self.name + '(' + str(self.arg) + ')'

    def copy(self) -> Formula:
        return Nary(self.name, self.arg)


class Term(Predicate, ABC):
    pass


class Constant(Term, ABC):
    pass


class Predication(Constant):

    def __init__(self, name: str):
        self.name: str = name

    def __str__(self) -> str:
        return self.name

    def copy(self) -> Formula:
        return Predication(self.name)


class Boolean(Constant):

    def __init__(self, value: bool = True):
        self.value: bool = value

    def __str__(self) -> str:
        return str(self.value)

    @property
    def is_true(self) -> bool:
        return self.value

    def copy(self) -> Formula:
        return Boolean(self.value)


class Number(Constant):

    def __init__(self, value: str):
        self.value: float = float(value)

    def __str__(self) -> str:
        return str(self.value)

    def copy(self) -> Formula:
        return Number(self.value)


class Variable(Term):

    def __init__(self, name: str):
        self.name: str = name

    def __str__(self) -> str:
        return self.name

    def copy(self) -> Formula:
        return Variable(self.name)


class Argument(Formula):

    def __init__(self, term: Term, arg: Argument = None):
        self.term: Term = term
        self.arg: Argument = arg

    def __str__(self) -> str:
        return str(self.term) + (',' + str(self.arg) if self.arg is not None else '')

    def copy(self) -> Formula:
        return Argument(self.term.copy(), self.arg)
