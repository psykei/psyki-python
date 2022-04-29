from abc import ABC, abstractmethod
from os.path import isdir
from typing import Any
from antlr4 import CommonTokenStream, InputStream
from psyki.logic.datalog.grammar import *
from psyki.resources import PATH, create_antlr4_parser
if not isdir(str(PATH / 'dist')):
    create_antlr4_parser(str(PATH / 'Datalog.g4'), str(PATH / 'dist'))
from psyki.resources.dist.DatalogParser import DatalogParser
from psyki.resources.dist.DatalogLexer import DatalogLexer


class Adapter(ABC):

    @abstractmethod
    def get_formula(self, ast: Any) -> DatalogFormula:
        pass

    @abstractmethod
    def get_formula_from_string(self, rule: str) -> DatalogFormula:
        pass


class Antlr4(Adapter):

    def __init__(self):
        pass

    def get_formula_from_string(self, rule: str) -> DatalogFormula:
        return self.get_formula(DatalogParser(CommonTokenStream(DatalogLexer(InputStream(rule)))).formula())

    def get_formula(self, ast: DatalogParser.FormulaContext) -> DatalogFormula:
        return DatalogFormula(self._get_definition_clause(ast.lhs), self._get_clause(ast.rhs), self._get_clause(ast.op))

    def _get_definition_clause(self, node: DatalogParser.DefPredicateArgsContext):
        return DefinitionClause(node.pred.text, self._get_arguments(node.args))

    def _get_arguments(self, node: DatalogParser.MoreArgsContext or DatalogParser.LastTermContext):
        if isinstance(node, DatalogParser.MoreArgsContext):
            return Argument(self._get_term(node.name), self._get_arguments(node.args))
        elif isinstance(node, DatalogParser.LastTermContext):
            return Argument(self._get_term(node.name))

    def _get_clause(self,
                    node: DatalogParser.ClauseExpressionContext or DatalogParser.ClauseExpressionNoParContext or
                          DatalogParser.ClauseLiteralContext or DatalogParser.ClauseClauseContext):
        if isinstance(node, DatalogParser.ClauseExpressionContext) \
                or isinstance(node, DatalogParser.ClauseExpressionNoParContext):
            return Expression(self._get_clause(node.left), self._get_clause(node.right), node.op.text)
        elif isinstance(node, DatalogParser.ClauseLiteralContext):
            return self._get_literal(node.lit)
        elif isinstance(node, DatalogParser.ClauseClauseContext):
            return self._get_clause(node.c)

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
            return Unary(node.pred.text)
        elif isinstance(node, DatalogParser.PredicateArgsContext):
            return Nary(node.pred.text, self._get_arguments(node.args))

    def _get_term(self, node: DatalogParser.TermVarContext or DatalogParser.TermConstContext):
        if isinstance(node, DatalogParser.TermVarContext):
            return Variable(node.var.text)
        elif isinstance(node, DatalogParser.TermConstContext):
            return self._get_constant(node.name)

    def _get_constant(self, node: DatalogParser.ConstNumberContext or DatalogParser.ConstNameContext):
        if isinstance(node, DatalogParser.ConstNumberContext):
            return Number(node.num.text)
        elif isinstance(node, DatalogParser.ConstNameContext):
            return Predication(node.name.text)
