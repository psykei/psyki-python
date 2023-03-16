from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterable
import psyki.logic.operators


PATH = Path(__file__).parents[0]


class TheoryAdapter(ABC):
    """
    Abstract adapter to convert a legacy logic theory into a Theory that can be used by injectors.
    """

    @staticmethod
    @abstractmethod
    def from_legacy_theory(legacy_theory: Any) -> Theory:
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def from_file(filename: str) -> Theory:
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def from_string(textual_theory: str) -> Theory:
        raise NotImplementedError()


class Theory:
    """
    Uniformized logic theory that can be used by injectors.
    """

    def __init__(self, formulae: list[Formula] = None):
        self.formulae: list[Formula] = formulae

    def __add__(self, other: Theory):
        self.formulae += other.formulae

    def __repr__(self) -> str:
        return repr(self.formulae)

    def __str__(self) -> str:
        return '\n'.join(str(f) for f in self.formulae)

    def __eq__(self, other: Theory):
        return all(f1 == f2 for f1, f2 in zip(self.formulae, other.formulae))

    def __hash__(self):
        raise hash(self.formulae)


class Formula(ABC):
    """
    Data structure that represents a logic formula.
    """

    @abstractmethod
    def copy(self) -> Formula:
        raise NotImplementedError()

    @abstractmethod
    def __repr__(self) -> str:
        raise NotImplementedError()

    @abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError()

    @abstractmethod
    def __eq__(self, other):
        raise NotImplementedError()

    @abstractmethod
    def __hash__(self):
        raise NotImplementedError()

    def optimize(self):
        optimize_formula(self)

    @property
    def is_optimized(self) -> bool:
        return False


def optimize_formula(formula: Formula) -> None:

    def optimize_child(child, operator, father):
        if isinstance(child, Expression):
            if child.op.name == operator.name:
                optimize_formula(child)
                for clause in child.unfolded_arguments:
                    father.unfolded_arguments.append(clause)
                # if father.lhs == child:
                #     father.lhs = None
                # elif father.rhs == child:
                #     father.rhs = None
            else:
                father.unfolded_arguments.append(child)
        else:
            father.unfolded_arguments.append(child)

    if isinstance(formula, Expression):
        lhs = formula.lhs
        rhs = formula.rhs
        op = formula.op
        if op.is_optimizable and len(formula.unfolded_arguments) == 0:
            optimize_child(lhs, op, formula)
            optimize_child(rhs, op, formula)
        else:
            optimize_formula(lhs)
            optimize_formula(rhs)
    else:
        if hasattr(formula, 'lhs'):
            optimize_formula(formula.lhs)
        if hasattr(formula, 'rhs'):
            optimize_formula(formula.rhs)
        if hasattr(formula, 'predicate'):
            optimize_formula(formula.predicate)


class DefinitionFormula(Formula):
    """
    Logic rule with a left-hand side with variables and possibly a term representing a class or a real value.
    Right-hand side contains the logic clauses that must be satisfied to make the whole rule true.
    """
    __definition_symbol: str = ':-'

    def __init__(self, lhs: DefinitionClause, rhs: Clause, trainable: bool = False):
        self.lhs: DefinitionClause = lhs
        self.rhs: Clause = rhs
        self.trainable = trainable

    def __str__(self) -> str:
        return str(self.lhs) + ' ' + self.__definition_symbol + ' ' + str(self.rhs)

    def __repr__(self) -> str:
        return repr(self.lhs) + self.__definition_symbol + repr(self.rhs)

    def __eq__(self, other: DefinitionFormula) -> bool:
        return self.lhs == other.lhs and self.lhs == other.rhs

    def __hash__(self) -> int:
        return hash((self.lhs, self.rhs))

    def copy(self) -> Formula:
        return DefinitionFormula(self.lhs.copy(), self.rhs.copy(), self.trainable)

    @property
    def arity(self) -> int:
        return self.lhs.arity

    @property
    def is_optimized(self) -> bool:
        return self.rhs.is_optimized

    def remove_variable_assignment(self, variables: Iterable[Variable]) -> DefinitionFormula:
        """
        Return a new formula without 'is' expressions in the body.
        If a variable's name appears in variable_names then the expression is substituted with the true predicate.
        If the variable's name does not appear in variable_names then 'is' is substituted with the equivalence.
        """
        return DefinitionFormula(self.lhs.copy(), self.rhs.remove_variable_assignment(variables), self.trainable)

    def get_substitution(self, variable: Variable) -> Formula:
        """
        Return the assigned formula to a specific variable.
        If there is no 'is' predicate for the provided variable return the variable itself.
        """
        return self.rhs.get_substitution(variable)


class DefinitionClause(Formula):
    """
    Left-hand side of a logic rule.
    """

    def __init__(self, predication: str, args: Argument):
        self.predication: str = predication
        self.args: Argument = args

    def __repr__(self) -> str:
        return self.predication + '(' + (repr(self.args)) + ')'

    def __str__(self) -> str:
        return self.predication + '(' + (str(self.args)) + ')'

    def __eq__(self, other: DefinitionClause) -> bool:
        return self.predication == other.predication and self.args == other.args

    def __hash__(self) -> int:
        return hash((self.predication, self.args))

    def copy(self) -> DefinitionClause:
        return DefinitionClause(self.predication, self.args)

    @property
    def arity(self) -> int:
        return len(self.args.unfolded)


class Clause(Formula, ABC):

    def remove_variable_assignment(self, variables: Iterable[Variable]) -> Clause:
        return self

    def get_substitution(self, variable: Variable) -> Formula:
        return variable


class Expression(Clause):
    """
    Logic expression with arity 2, so it is composed of a left-hand side and a right-hand side plus the operator.
    """
    def __init__(self, lhs: Clause, rhs: Clause, op: operators.LogicOperator):
        self.lhs: Clause = lhs
        self.rhs: Clause = rhs
        self.unfolded_arguments: list[Clause] = []
        self.op: operators.LogicOperator = op

    def __repr__(self) -> str:
        if len(self.unfolded_arguments) > 0:
            return "'" + repr(self.op) + "'(" + ','.join(repr(arg) for arg in self.unfolded_arguments) + ")"
        else:
            return repr(self.lhs) + repr(self.op) + repr(self.rhs)

    def __str__(self) -> str:
        if len(self.unfolded_arguments) > 0:
            return "'" + str(self.op) + "'(" + ','.join(str(arg) for arg in self.unfolded_arguments) + ")"
        else:
            return str(self.lhs) + self.op.pretty_string + str(self.rhs)

    def __eq__(self, other: Expression) -> bool:
        return self.lhs == other.lhs and self.rhs == other.rhs and self.op is other.op

    def __hash__(self) -> int:
        return hash((self.lhs, self.rhs, self.op))

    def copy(self) -> Expression:
        lhs = self.lhs.copy()
        rhs = self.rhs.copy()
        return Expression(lhs, rhs, self.op)

    @property
    def is_optimized(self) -> bool:
        if self.op.is_optimizable and len(self.unfolded_arguments) > 0:
            return True
        elif self.lhs.is_optimized or self.rhs.is_optimized:
            return True
        else:
            return False

    def remove_variable_assignment(self, variables: Iterable[Variable]) -> Clause:
        if self.op.symbol == operators.Assignment.symbol:
            assert isinstance(self.lhs, Variable)
            if self.lhs in variables:
                return Boolean(True)
            else:
                return Expression(self.lhs.copy(), self.rhs.copy(), operators.Equal())
        else:
            return Expression(self.lhs.remove_variable_assignment(variables),
                              self.rhs.remove_variable_assignment(variables), self.op)

    def get_substitution(self, variable: Variable) -> Formula:
        if isinstance(self.lhs, Variable) and self.lhs == variable and self.op.symbol == operators.Assignment.symbol:
            return self.rhs
        else:
            rhs = self.rhs.get_substitution(variable)
            lhs = self.lhs.get_substitution(variable)
            if isinstance(lhs, Variable) and lhs.name == variable.name:
                return rhs
            else:
                return lhs


class Literal(Clause, ABC):
    pass


class Negation(Literal):
    """
    Negation of a predicate.
    """
    __negation_operator: operators.LogicOperator = operators.LogicNegation()

    def __init__(self, predicate: Clause):
        self.predicate: Clause = predicate

    def __repr__(self) -> str:
        return self.__negation_operator.symbol + '(' + repr(self.predicate) + ')'

    def __str__(self) -> str:
        return self.__negation_operator.pretty_string + '(' + str(self.predicate) + ')'

    def __eq__(self, other: Negation) -> bool:
        return self.predicate == other.predicate

    def __hash__(self) -> int:
        return hash((self.__negation_operator, self.predicate))

    def copy(self) -> Negation:
        return Negation(self.predicate.copy())


class Predicate(Clause, ABC):
    pass


class Unary(Predicate):
    """
    Fact with one term.
    """

    def __init__(self, predicate: str, term: Term):
        self.predicate: str = predicate
        self.term: Term = term

    def __repr__(self) -> str:
        return repr(self.predicate) + "(" + repr(self.term) + ")"

    def __str__(self) -> str:
        return str(self.predicate) + "(" + str(self.term) + ")"

    def __eq__(self, other: Unary) -> bool:
        return self.predicate == other.predicate and self.term == self.term

    def __hash__(self) -> int:
        return hash((self.predicate, self.term))

    def copy(self) -> Unary:
        return Unary(self.predicate, self.term)


class Nary(Predicate):
    """
    Fact with multiple terms.
    """

    def __init__(self, predicate: str, args: Argument):
        self.predicate: str = predicate
        self.args: Argument = args

    def __repr__(self):
        return repr(self.predicate) + '(' + repr(self.args) + ')'

    def __str__(self) -> str:
        return self.predicate + '(' + str(self.args) + ')'

    def __eq__(self, other: Nary) -> bool:
        return self.predicate == other.predicate and self.arity == other.arity

    def __hash__(self) -> int:
        return hash((self.predicate, self.args))

    def copy(self) -> Nary:
        return Nary(self.predicate, self.args)

    @property
    def arity(self) -> int:
        return len(self.args.unfolded)


class Term(Predicate, ABC):
    pass


class Constant(Term, ABC):
    pass


class Predication(Constant):
    """
    Constant, usually it refers to a class for a classification task.
    """

    def __init__(self, name: str):
        self.name: str = name

    def __repr__(self) -> str:
        return repr(self.name)

    def __str__(self) -> str:
        return self.name

    def __eq__(self, other: Predication) -> bool:
        return self.name == other.name

    def __hash__(self) -> int:
        return hash(self.name)

    def copy(self) -> Predication:
        return Predication(self.name)


class Boolean(Constant):
    """
    Boolean value.
    """

    def __init__(self, value: bool = True):
        self.value: bool = value

    def __repr__(self) -> str:
        return repr(self.value)

    def __str__(self) -> str:
        return str(self.value)

    def __eq__(self, other: Boolean) -> bool:
        return self.value == other.value

    def __hash__(self) -> int:
        return hash(self.value)

    @property
    def is_true(self) -> bool:
        return self.value

    def copy(self) -> Boolean:
        return Boolean(self.value)


class Number(Constant):
    """
    Real number, usually it refers to the output value of a regression task or to value of features.
    """

    def __init__(self, value: str):
        self.value: float = float(value)

    def __repr__(self) -> str:
        return repr(self.value)

    def __str__(self) -> str:
        return str(self.value)

    def __eq__(self, other: Number) -> bool:
        return False if not isinstance(other, Number) else self.value == other.value

    def __hash__(self) -> int:
        return hash(self.value)

    def copy(self) -> Number:
        return Number(str(self.value))


class Variable(Term):
    """
    Logic variable, usually it refers to one feature of the ML task.
    """

    def __init__(self, name: str):
        self.name: str = name

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.name

    def __eq__(self, other: Variable) -> bool:
        return self.name == other.name

    def __hash__(self) -> int:
        return hash(self.name)

    def copy(self) -> Variable:
        return Variable(self.name)


class Argument(Formula):
    """
    One argument of a predicate.
    """

    def __init__(self, term: Term, args: Argument = None):
        self.term: Term = term
        self.args: Argument = args

    def __repr__(self) -> str:
        return repr(self.term) + (',' + repr(self.args) if self.args is not None else '')

    def __str__(self) -> str:
        return str(self.term) + (', ' + str(self.args) if self.args is not None else '')

    def __eq__(self, other: Argument) -> bool:
        return self.term == other.term and self.args == other.args

    def __hash__(self) -> int:
        return hash((self.term, self.args))

    def copy(self) -> Argument:
        return Argument(self.term.copy(), self.args)

    @property
    def unfolded(self) -> list[Term]:
        if self.args is None:
            return [self.term]
        else:
            return [self.term] + self.args.unfolded

    @property
    def last(self) -> Term:
        return self.unfolded[-1]


class ComplexArgument(Formula):
    """
    One argument of a predicate.
    """

    def __init__(self, clause: Clause, args: ComplexArgument = None):
        self.clause: Clause = clause
        self.args: ComplexArgument = args

    def __repr__(self) -> str:
        return str(self.clause) + (',' + str(self.args) if self.args is not None else '')

    def __str__(self) -> str:
        return str(self.clause) + (',' + str(self.args) if self.args is not None else '')

    def __eq__(self, other: ComplexArgument) -> bool:
        return self.clause == other.clause and self.args == other.args

    def __hash__(self) -> int:
        return hash((self.clause, self.args))

    def copy(self) -> ComplexArgument:
        return ComplexArgument(self.clause.copy(), self.args)

    @property
    def unfolded(self):
        if self.args is None:
            return [self.clause]
        else:
            return [self.clause] + self.args.unfolded

    @property
    def last(self):
        return self.unfolded[-1]
