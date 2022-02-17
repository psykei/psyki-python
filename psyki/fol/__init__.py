from __future__ import annotations
from abc import ABC
from typing import Set, List, TypeVar
from tuprolog.core import Var, Term

T = TypeVar('T')

class FOLFormula(ABC):

    quantifiers: Quantification = None
    clause: Clause = None


class Quantifier(ABC):

    variable: Var


class ForAll(Quantifier):
    pass


class Exist(Quantifier):
    pass


class Quantification(Set[Quantifier]):
    pass


class Clause(ABC):
    pass


class Literal(Clause, ABC):

    negative: bool = None
    predicate: Predicate = None


class Predicate(ABC):

    predication: str = None
    arguments: List[Term] = None


class Expression(Clause, ABC):

    connective: str = None
    arguments: List[FOLFormula] = None
