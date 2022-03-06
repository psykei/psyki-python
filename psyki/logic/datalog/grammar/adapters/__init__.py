from abc import ABC, abstractmethod
from typing import Any
from psyki.logic.datalog.grammar import DatalogFormula, Expression, Predicate, Literal, Variable, Number, Unary
from psyki.ski import Formula
from resources.dist.resources.DatalogParser import DatalogParser


class Adapter(ABC):

    @abstractmethod
    def get_formula(self, ast: Any) -> DatalogFormula:
        pass


class Antlr4(Adapter):

    mapping: dict[DatalogParser.__class__: Formula.__class__] = {
        DatalogParser.FormulaContext: DatalogFormula,
        DatalogParser.ClauseExpressionNoParContext: Expression,
        DatalogParser.ClauseExpressionContext: Expression,
        DatalogParser.LiteralContext: Literal,
        DatalogParser.PredicateContext: Predicate,
        DatalogParser.TermVarContext: Variable,
        DatalogParser.ConstNumberContext: Number,
        DatalogParser.ConstNameContext: Unary,
    }

    def __init__(self):
        pass

    def get_formula(self, ast: Any) -> DatalogFormula:
        pass


"""
        def __init__(self, tree: Any):
            self.logic_class = self.mapping.get(tree.__class__)
            self.value = tree.symbol.text if hasattr(tree, 'symbol') else None
            self.children: list[FOLTree] = [FOLTree(child) for child in tree.children
                                            if not hasattr(child, 'symbol') or child.symbol.text not in ('(', ')')] \
                if hasattr(tree, 'children') else []

        def element(self, element: Any) -> FOLTree:
            if self.logic_class == element:
                return self
            else:
                candidates = [child.element(element) for child in self.children if child.element(element) is not None]
                return candidates[0] if len(candidates) > 0 else None

        def elements(self, element: Any) -> list[FOLTree]:
            if self.logic_class == element:
                return [self]
            else:
                return [child.element(element) for child in self.children if child.element(element) is not None]
"""