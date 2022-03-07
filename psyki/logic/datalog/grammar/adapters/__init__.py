from abc import ABC, abstractmethod
from typing import Any
from psyki.logic.datalog.grammar import DatalogFormula, Expression, DefinitionClause, Argument, Negation, Unary, Nary, \
    Variable, Number, Predication
from resources.dist.resources.DatalogParser import DatalogParser


class Adapter(ABC):

    @abstractmethod
    def get_formula(self, ast: Any) -> DatalogFormula:
        pass


class Antlr4(Adapter):

    def __init__(self):
        pass

    """def _visit(self, node: Formula):
        self.mapping.get(node.__class__)(node)"""

    def get_formula(self, ast: DatalogParser.FormulaContext) -> DatalogFormula:
        return DatalogFormula(self._get_definition_clause(ast.lhs), self._get_clause(ast.rhs), '←')

    def _get_definition_clause(self, node: DatalogParser.DefPredicateArgsContext):
        return DefinitionClause(node.pred, self._get_arguments(node.args))

    def _get_arguments(self, node: DatalogParser.MoreArgsContext or DatalogParser.LastTermContext):
        if isinstance(node, DatalogParser.MoreArgsContext):
            return Argument(node.name, self._get_arguments(node.args))
        elif isinstance(node, DatalogParser.LastTermContext):
            return Argument(node.name)

    def _get_clause(self,
                    node: DatalogParser.ClauseExpressionContext or DatalogParser.ClauseExpressionNoParContext or DatalogParser.ClauseLiteralContext):
        if isinstance(node, DatalogParser.ClauseExpressionContext) \
                or isinstance(node, DatalogParser.ClauseExpressionNoParContext):
            return Expression(self._get_clause(node.left), self._get_clause(node.right), node.op)
        elif isinstance(node, DatalogParser.ClauseLiteralContext):
            return self._get_literal(node.lit)

    def _get_literal(self, node: DatalogParser.LiteralPredContext or DatalogParser.LiteralNegContext):
        if isinstance(node, DatalogParser.LiteralNegContext):
            return Negation(self._get_clause(node.pred))
        elif isinstance(node, DatalogParser.LiteralPredContext):
            return self._get_predicate(node.pred)

    def _get_predicate(self, node: DatalogParser.PredicateTermContext or DatalogParser.PredicateUnaryContext
                                   or DatalogParser.PredicateArgsContext):
        if isinstance(node, DatalogParser.PredicateTermContext):
            return self._get_term(node.name)
        elif isinstance(node, DatalogParser.PredicateUnaryContext):
            return Unary(node.pred)
        elif isinstance(node, DatalogParser.PredicateArgsContext):
            return Nary(node.pred, self._get_arguments(node.args))

    def _get_term(self, node: DatalogParser.TermVarContext or DatalogParser.TermConstContext):
        if isinstance(node, DatalogParser.TermVarContext):
            return Variable(node.var)
        elif isinstance(node, DatalogParser.TermConstContext):
            return self._get_constant(node.name)

    def _get_constant(self, node: DatalogParser.ConstNumberContext or DatalogParser.ConstNameContext):
        if isinstance(node, DatalogParser.ConstNumberContext):
            return Number(node.num)
        elif isinstance(node, DatalogParser.ConstNameContext):
            return Predication(node.name)
