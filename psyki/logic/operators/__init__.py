from __future__ import annotations
from abc import abstractmethod, ABCMeta, ABC
from pathlib import Path


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
        match symbol:
            case LogicNegation.symbol:
                return LogicNegation()
            case Conjunction.symbol:
                return Conjunction()
            case Disjunction.symbol:
                return Disjunction()
            case Assignment.symbol:
                return Assignment()
            case Equal.symbol:
                return Equal()
            case Greater.symbol:
                return Greater()
            case Less.symbol:
                return Less()
            case GreaterEqual.symbol:
                return GreaterEqual()
            case LessEqual.symbol:
                return LessEqual()
            case Plus.symbol:
                return Plus()
            case Multiplication.symbol:
                return Multiplication()
            case _:
                return None  # raise Exception("Unexpected type")


class BinaryOperator(ABC, LogicOperator):
    arity: int = 2


class UnaryOperator(ABC, LogicOperator):
    arity: int = 1


class LogicNegation(UnaryOperator):
    name = "negation"
    symbol = r"\+"

    def pretty_string(self) -> str:
        return self.symbol + ' '


class Conjunction(BinaryOperator):
    name = "conjunction"
    symbol = ","

    def pretty_string(self) -> str:
        return self.symbol + ' '


class Disjunction(BinaryOperator):
    name = "disjunction"
    symbol = ";"

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

    def pretty_string(self) -> str:
        return ' ' + self.symbol + ' '


class LessEqual(BinaryOperator):
    name = "less equal"
    symbol = "=<"

    def pretty_string(self) -> str:
        return ' ' + self.symbol + ' '


class Less(BinaryOperator):
    name = "less"
    symbol = "<"

    def pretty_string(self) -> str:
        return ' ' + self.symbol + ' '


class Greater(BinaryOperator):
    name = "greater"
    symbol = ">"

    def pretty_string(self) -> str:
        return ' ' + self.symbol + ' '


class GreaterEqual(BinaryOperator):
    name = "greater equal"
    symbol = ">="

    def pretty_string(self) -> str:
        return ' ' + self.symbol + ' '


class Plus(BinaryOperator):
    name = "plus"
    symbol = "+"

    def pretty_string(self) -> str:
        return ' ' + self.symbol + ' '


class Multiplication(BinaryOperator):
    name = "multiplication"
    symbol = "*"

    def pretty_string(self) -> str:
        return ' ' + self.symbol + ' '
