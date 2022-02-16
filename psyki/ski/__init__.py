from tensorflow import maximum, minimum
from psyki.utils import eta
from resources.dist.resources.PrologParser import PrologParser
from resources.dist.resources.PrologVisitor import PrologVisitor


class Fuzzifier(PrologVisitor):

    def __init__(self, feature_mapping: dict[str, int]):
        self.feature_mapping = feature_mapping

    # Visit a parse tree produced by folParser#formula.
    def visitFormula(self, ctx: PrologParser.FormulaContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by folParser#ClauseLiteral.
    def visitClauseLiteral(self, ctx: PrologParser.ClauseLiteralContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by folParser#ClauseExpression.
    def visitClauseExpression(self, ctx: PrologParser.ClauseExpressionContext):
        l, r = self.visit(ctx.left), self.visit(ctx.right)
        operation = {
            '∧': lambda x: eta(maximum(l(x), r(x))),
            '∨': lambda x: eta(minimum(l(x), r(x))),
            '→': lambda x: eta(l(x) - r(x)),
            '↔': lambda x: eta(abs(l(x) - r(x))),
            '=': lambda x: eta(abs(l(x) - r(x))),
            '<': lambda x: eta(1. - eta(1. - eta(l(x) - r(x)))),
            '≤': lambda x: eta(1. - eta(maximum(eta(1. - eta(l(x) - r(x))), eta(1. - eta(abs(l(x) - r(x))))))),
            '>': lambda x: eta(maximum(eta(1. - eta(l(x) - r(x))), eta(1. - eta(abs(l(x) - r(x)))))),
            '≥': lambda x: eta(1. - eta(l(x) - r(x)))
        }
        return operation.get(ctx.op.text)

    # Visit a parse tree produced by PrologParser#ConstNumber.
    def visitConstNumber(self, ctx: PrologParser.ConstNumberContext):
        return lambda _: float(ctx.num.text)

    # Visit a parse tree produced by folParser#TermVar.
    def visitTermVar(self, ctx: PrologParser.TermVarContext):
        var = ctx.var.text
        return lambda x: x[:, self.feature_mapping[var]]
