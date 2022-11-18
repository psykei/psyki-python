from os.path import isdir
from psyki.logic.datalog.grammar import DatalogFormula, Expression, DefinitionClause, Argument, Negation, Unary, Nary, \
    Variable, Number, Predication, MofN, ComplexArgument, optimize_datalog_formula
from psyki.resources import PATH, create_antlr4_parser
if not isdir(str(PATH / 'dist')):
    create_antlr4_parser(str(PATH / 'Datalog.g4'), str(PATH / 'dist'))
from antlr4 import CommonTokenStream, InputStream
from psyki.resources.dist.DatalogLexer import DatalogLexer
from psyki.resources.dist.DatalogParser import DatalogParser


def get_formula_from_string(rule: str) -> DatalogFormula:
    return get_formula(DatalogParser(CommonTokenStream(DatalogLexer(InputStream(rule)))).formula())


def get_formula(ast: DatalogParser.FormulaContext) -> DatalogFormula:
    formula: DatalogFormula = DatalogFormula(_get_definition_clause(ast.lhs), _get_clause(ast.rhs), ast.op.text)
    optimize_datalog_formula(formula)
    return formula


def to_prolog_string(rule: DatalogFormula) -> str:
    # TODO: for now it works with just propositional rules without clauses except 'class'.

    def unfold_body(e: Expression, result=[]) -> list[str]:
        if isinstance(e.lhs, Expression):
            result = result + unfold_body(e.lhs)
        if isinstance(e.rhs, Expression):
            result = result + unfold_body(e.rhs)
        if not isinstance(e.lhs, Expression) and not isinstance(e.rhs, Expression):
            return ['\'' + e.op + '\'(' + str(e.lhs) + ', ' + str(e.rhs) + ')']
        return result

    head = str(rule.lhs)
    if isinstance(rule.rhs, Expression):
        body = ', '.join(unfold_body(rule.rhs))
        return head + ' :- ' + body + '.'
    else:
        return head + '.'


def _get_definition_clause(node: DatalogParser.DefPredicateArgsContext):
    return DefinitionClause(node.pred.text, _get_arguments(node.args))


def _get_arguments(node: DatalogParser.MoreArgsContext or DatalogParser.LastTermContext):
    if isinstance(node, DatalogParser.MoreArgsContext):
        return Argument(_get_term(node.name), _get_arguments(node.args))
    elif isinstance(node, DatalogParser.LastTermContext):
        return Argument(_get_term(node.name))


def _get_complex_arguments(node: DatalogParser.MoreComplexArgsContext or DatalogParser.LastClauseContext):
    if isinstance(node, DatalogParser.MoreComplexArgsContext):
        return ComplexArgument(_get_clause(node.name), _get_complex_arguments(node.args))
    elif isinstance(node, DatalogParser.LastClauseContext):
        return ComplexArgument(_get_clause(node.name))


def _get_clause(node: DatalogParser.ClauseExpressionContext or DatalogParser.ClauseExpressionNoParContext
                      or DatalogParser.ClauseLiteralContext or DatalogParser.ClauseClauseContext):
    if isinstance(node, DatalogParser.ClauseExpressionContext)\
            or isinstance(node, DatalogParser.ClauseExpressionNoParContext):
        return Expression(_get_clause(node.left), _get_clause(node.right), node.op.text)
    elif isinstance(node, DatalogParser.ClauseLiteralContext):
        return _get_literal(node.lit)
    elif isinstance(node, DatalogParser.ClauseClauseContext):
        return _get_clause(node.c)


def _get_literal(node: DatalogParser.LiteralPredContext or DatalogParser.LiteralNegContext):
    if isinstance(node, DatalogParser.LiteralNegContext):
        return Negation(_get_clause(node.pred))
    elif isinstance(node, DatalogParser.LiteralPredContext):
        return _get_predicate(node.pred)


def _get_predicate(node: DatalogParser.PredicateTermContext or DatalogParser.PredicateUnaryContext
                         or DatalogParser.PredicateArgsContext):
    if isinstance(node, DatalogParser.PredicateTermContext):
        return _get_term(node.name)
    elif isinstance(node, DatalogParser.PredicateUnaryContext):
        return Unary(node.pred.text)
    elif isinstance(node, DatalogParser.MofNContext):
        return MofN(node.pred.text, node.m.text, _get_complex_arguments(node.args))
    elif isinstance(node, DatalogParser.PredicateArgsContext):
        return Nary(node.pred.text, _get_arguments(node.args))


def _get_term(node: DatalogParser.TermVarContext or DatalogParser.TermConstContext):
    if isinstance(node, DatalogParser.TermVarContext):
        return Variable(node.var.text)
    elif isinstance(node, DatalogParser.TermConstContext):
        return _get_constant(node.name)


def _get_constant(node: DatalogParser.ConstNumberContext or DatalogParser.ConstNameContext):
    if isinstance(node, DatalogParser.ConstNumberContext):
        return Number(node.num.text)
    elif isinstance(node, DatalogParser.ConstNameContext):
        return Predication(node.name.text)
