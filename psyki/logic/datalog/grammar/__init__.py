from __future__ import annotations
from abc import ABC
from typing import Iterable
from psyki.logic import Formula
import psyki.logic.datalog as datalog


# TODO: refactoring
def optimize_datalog_formula(formula: Formula) -> None:
    if isinstance(formula, datalog.grammar.Expression):
        lhs = formula.lhs
        rhs = formula.rhs
        op = formula.op
        if op in (',', ';', '+') and len(formula.nary) == 0:
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

    def __init__(self, lhs: DefinitionClause, rhs: Clause, op: str = '<-'):
        self.lhs: DefinitionClause = lhs
        self.rhs: Clause = rhs
        self.op: str = op

    def __str__(self) -> str:
        return str(self.lhs) + ' ' + self.op + ' ' + str(self.rhs)

    def __repr__(self) -> str:
        return repr(self.lhs) + self.op + repr(self.rhs)

    def copy(self) -> Formula:
        return DatalogFormula(self.lhs.copy(), self.rhs.copy(), self.op)


class DefinitionClause(Formula):

    def __init__(self, predication: str, arg: Argument):
        self.predication: str = predication
        self.arg: Argument = arg

    def __repr__(self) -> str:
        return self.predication + '(' + (repr(self.arg) if self.arg is not None else '') + ')'

    def __str__(self) -> str:
        return self.predication + '(' + (str(self.arg) if self.arg is not None else '') + ')'

    def copy(self) -> DefinitionClause:
        return DefinitionClause(self.predication, self.arg)


class Clause(Formula, ABC):

    def copy(self) -> Clause:
        pass


class Expression(Clause):

    def __init__(self, lhs: Clause, rhs: Clause, op: str, nary: Iterable[Clause] = None):
        self.lhs: Clause = lhs
        self.rhs: Clause = rhs
        self.nary: list[Clause] = list(nary) if nary is not None else []
        self.op: str = op

    def __repr__(self) -> str:
        if len(self.nary) <= 2:
            return repr(self.lhs) + repr(self.op) + repr(self.rhs)
        else:
            return "'" + self.op + "'(" + ','.join(repr(clause) for clause in self.nary) + ')'

    def __str__(self) -> str:
        if len(self.nary) <= 2:
            return str(self.lhs) + ('' if self.op == ',' else ' ') + str(self.op) + ' ' + str(self.rhs)
        else:
            return "'" + self.op + "'(" + ', '.join(str(clause) for clause in self.nary) + ')'

    def copy(self) -> Expression:
        lhs = self.lhs.copy() if self.lhs is not None else None
        rhs = self.rhs.copy() if self.rhs is not None else None
        return Expression(lhs, rhs, self.op, [c.copy() for c in self.nary if c is not None])


class Literal(Clause, ABC):
    pass


class Negation(Literal):

    def __init__(self, predicate: Clause):
        self.predicate: Clause = predicate

    def __repr__(self) -> str:
        return 'not(' + repr(self.predicate) + ')'

    def __str__(self) -> str:
        return 'not(' + str(self.predicate) + ')'

    def copy(self) -> Negation:
        return Negation(self.predicate.copy())


class Predicate(Clause, ABC):

    name: str = ''

    @property
    def _name(self) -> str:
        return self.name if self.name[0].islower() else "'" + self.name + "'"


class Unary(Predicate):

    def __init__(self, name: str):
        self.name: str = name

    def __repr__(self) -> str:
        return repr(self.name) + "()"

    def __str__(self) -> str:
        return self._name + "()"

    def copy(self) -> Unary:
        return Unary(self.name)


class MofN(Predicate):
    def __init__(self, name: str, m: Number, arg: ComplexArgument):
        self.name: str = name
        self.m: Number = m
        self.arg: ComplexArgument = arg

    def __repr__(self) -> str:
        return repr(self.name) + '(' + repr(self.m) + ',' + repr(self.arg) + ')'

    def __str__(self) -> str:
        return self._name + '(' + str(self.m) + ', ' + str(self.arg) + ')'

    def copy(self) -> MofN:
        return MofN(self.name, self.m, self.arg)


class Nary(Predicate):

    def __init__(self, name: str, arg: Argument):
        self.name: str = name
        self.arg: Argument = arg

    def __repr__(self):
        return repr(self.name) + '(' + (repr(self.arg) if self.arg is not None else '') + ')'

    def __str__(self) -> str:
        return self._name + '(' + (str(self.arg) if self.arg is not None else '') + ')'

    def copy(self) -> Nary:
        return Nary(self.name, self.arg)


class Term(Predicate, ABC):
    pass


class Constant(Term, ABC):
    pass


class Predication(Constant):

    def __init__(self, name: str):
        self.name: str = name

    def __repr__(self) -> str:
        return repr(self.name)

    def __str__(self) -> str:
        return self._name

    def copy(self) -> Predication:
        return Predication(self.name)


class Boolean(Constant):

    def __init__(self, value: bool = True):
        self.value: bool = value

    def __repr__(self) -> str:
        return repr(self.value)

    def __str__(self) -> str:
        return str(self.value)

    @property
    def is_true(self) -> bool:
        return self.value

    def copy(self) -> Boolean:
        return Boolean(self.value)


class Number(Constant):

    def __init__(self, value: str):
        self.value: float = float(value)

    def __repr__(self) -> str:
        return repr(self.value)

    def __str__(self) -> str:
        return str(self.value)

    def copy(self) -> Number:
        return Number(str(self.value))


class Variable(Term):

    def __init__(self, name: str):
        self.name: str = name

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.name

    def copy(self) -> Variable:
        return Variable(self.name)


class Argument(Formula):

    def __init__(self, term: Term, arg: Argument = None):
        self.term: Term = term
        self.arg: Argument = arg

    def __repr__(self) -> str:
        return str(self.term) + (',' + str(self.arg) if self.arg is not None else '')

    def __str__(self) -> str:
        return str(self.term) + (',' + str(self.arg) if self.arg is not None else '')

    def copy(self) -> Argument:
        return Argument(self.term.copy(), self.arg)

    @property
    def unfolded(self) -> list[Term]:
        if self.arg is None:
            return [self.term]
        else:
            return [self.term] + self.arg.unfolded

    @property
    def last(self) -> Term:
        return self.unfolded[-1]


class ComplexArgument(Formula):

    def __init__(self, clause: Clause, arg: ComplexArgument = None):
        self.clause: Clause = clause
        self.arg: ComplexArgument = arg

    def __repr__(self) -> str:
        return str(self.clause) + (',' + str(self.arg) if self.arg is not None else '')

    def __str__(self) -> str:
        return str(self.clause) + (',' + str(self.arg) if self.arg is not None else '')

    def copy(self) -> ComplexArgument:
        return ComplexArgument(self.clause.copy(), self.arg)

    @property
    def unfolded(self):
        if self.arg is None:
            return [self.clause]
        else:
            return [self.clause] + self.arg.unfolded

    @property
    def last(self):
        return self.unfolded[-1]
