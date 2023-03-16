from __future__ import annotations
from abc import abstractmethod, ABCMeta, ABC
from pathlib import Path
from psyki.utils import match_case


PATH = Path(__file__).parents[0]


class SingletonABCMeta(ABCMeta):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(SingletonABCMeta, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class LogicOperator(object):
    name: str
    symbol: str
    arity: int
    is_optimizable: bool = False
    __metaclass__ = SingletonABCMeta

    def __repr__(self) -> str:
        return self.symbol

    def __str__(self) -> str:
        return self.name

    @property
    @abstractmethod
    def pretty_string(self) -> str:
        raise NotImplementedError()

    @staticmethod
    def from_symbol(symbol: str) -> LogicOperator:
        cases = [
            (LogicNegation.symbol, LogicNegation),
            (Conjunction.symbol, Conjunction),
            (Disjunction.symbol, Disjunction),
            (Assignment.symbol, Assignment),
            (Equal.symbol, Equal),
            (GreaterEqual.symbol, GreaterEqual),
            (Greater.symbol, Greater),
            (LessEqual.symbol, LessEqual),
            (Less.symbol, Less),
            (Plus.symbol, Plus),
            (Multiplication.symbol, Multiplication),
            (symbol, None)
        ]
        matched = match_case(symbol, cases)
        return matched() if matched is not None else None


class Optimizable(ABC, LogicOperator):
    is_optimizable: bool = True


class BinaryOperator(ABC, LogicOperator):
    arity: int = 2


class UnaryOperator(ABC, LogicOperator):
    arity: int = 1


class LogicNegation(UnaryOperator):
    name = "negation"
    symbol = r"\+"

    @property
    def pretty_string(self) -> str:
        return self.symbol + ' '


class Conjunction(BinaryOperator, Optimizable):
    name = "conjunction"
    symbol = ","

    @property
    def pretty_string(self) -> str:
        return self.symbol + ' '


class Disjunction(BinaryOperator, Optimizable):
    name = "disjunction"
    symbol = ";"

    @property
    def pretty_string(self) -> str:
        return self.symbol + ' '


class Assignment(BinaryOperator):
    name = "assignment"
    symbol = 'is'

    @property
    def pretty_string(self) -> str:
        return ' ' + self.symbol + ' '


class Equal(BinaryOperator):
    name = "equal"
    symbol = "="

    @property
    def pretty_string(self) -> str:
        return ' ' + self.symbol + ' '


class LessEqual(BinaryOperator):
    name = "less equal"
    symbol = "=<"

    @property
    def pretty_string(self) -> str:
        return ' ' + self.symbol + ' '


class Less(BinaryOperator):
    name = "less"
    symbol = "<"

    @property
    def pretty_string(self) -> str:
        return ' ' + self.symbol + ' '


class Greater(BinaryOperator):
    name = "greater"
    symbol = ">"

    @property
    def pretty_string(self) -> str:
        return ' ' + self.symbol + ' '


class GreaterEqual(BinaryOperator):
    name = "greater equal"
    symbol = ">="

    @property
    def pretty_string(self) -> str:
        return ' ' + self.symbol + ' '


class Plus(BinaryOperator, Optimizable):
    name = "plus"
    symbol = "+"

    @property
    def pretty_string(self) -> str:
        return ' ' + self.symbol + ' '


class Multiplication(BinaryOperator, Optimizable):
    name = "multiplication"
    symbol = "*"

    @property
    def pretty_string(self) -> str:
        return ' ' + self.symbol + ' '
