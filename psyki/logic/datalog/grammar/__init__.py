from __future__ import annotations
from abc import ABC
from psyki.ski import Formula


class DatalogFormula(Formula):

    def __init__(self, lhs: DefinitionClause, rhs: Clause, op: str):
        self.lhs: DefinitionClause = lhs
        self.rhs: Clause = rhs
        self.op: str = op


class DefinitionClause(Formula):

    def __init__(self, predication: str, arg: Argument):
        self.predication: str = predication
        self.arg: Argument = arg


class Clause(Formula, ABC):
    pass


class Expression(Clause):

    def __init__(self, lhs: Clause, rhs: Clause, op: str):
        self.lhs: Clause = lhs
        self.rhs: Clause = rhs
        self.op: str = op


class Literal(Clause, ABC):
    pass


class Negation(Literal):

    def __init__(self, predicate: Clause):
        self.predicate: Clause = predicate


class Predicate(Clause, ABC):
    pass


class Unary(Predicate):

    def __init__(self, name: str):
        self.name: str = name


class Nary(Predicate):

    def __init__(self, name: str, arg: Argument):
        self.name: str = name
        self.arg: Argument = arg


class Term(Predicate, ABC):
    pass


class Constant(Term, ABC):
    pass


class Predication(Constant):

    def __init__(self, name: str):
        self.name: str = name


class Number(Constant):

    def __init__(self, value: str):
        self.value: float = float(value)


class Variable(Term):

    def __init__(self, name: str):
        self.name: str = name


class Argument(Formula):

    def __init__(self, term: Term, arg: Argument = None):
        self.term: Term = term
        self.arg: Argument = arg