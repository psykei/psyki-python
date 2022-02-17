from abc import ABC, abstractmethod
from typing import Any, Iterable
from antlr4 import CommonTokenStream, InputStream
from tensorflow import maximum, minimum, stack, Tensor, reduce_max, SparseTensor, cast, tile, reshape, shape
from tensorflow.keras.layers import Concatenate, Lambda
from tensorflow.keras import Model
from tensorflow.keras.backend import to_dense
from psyki.utils import eta
from resources.dist.resources.PrologLexer import PrologLexer
from resources.dist.resources.PrologParser import PrologParser
from resources.dist.resources.PrologVisitor import PrologVisitor


class Fuzzifier(PrologVisitor):

    def __init__(self, class_mapping: dict[str, int], feature_mapping: dict[str, int]):
        self.feature_mapping = feature_mapping
        self.class_mapping = {string: cast(to_dense(SparseTensor([[0, index]], [1.], (1, len(class_mapping)))), float)
                              for string, index in class_mapping.items()}

    # Visit a parse tree produced by folParser#formula.
    def visitFormula(self, ctx: PrologParser.FormulaContext):
        l = lambda y: eta(reduce_max(abs(tile(self.visit(ctx.args), (shape(y)[0], 1)) - y), axis=1))
        r = self.visit(ctx.right)
        return lambda x, y: eta(l(y) - r(x))

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

    # Visit a parse tree produced by PrologParser#ConstFunctor.
    def visitConstFunctor(self, ctx: PrologParser.ConstFunctorContext):
        return reshape(self.class_mapping[ctx.fun.text], (1, len(self.class_mapping)))

    # Visit a parse tree produced by PrologParser#ConstNumber.
    def visitConstNumber(self, ctx: PrologParser.ConstNumberContext):
        return lambda _: float(ctx.num.text)

    # Visit a parse tree produced by folParser#TermVar.
    def visitTermVar(self, ctx: PrologParser.TermVarContext):
        var = ctx.var.text
        return lambda x: x[:, self.feature_mapping[var]]


class Injector(ABC):
    """
    An injector is a class that allows a sub-symbolic predictor to exploit prior symbolic knowledge.
    The knowledge is provided via rules in some sort of logic form (e.g. FOL, Skolem, Horn).
    """
    predictor: Any  # Any class that has methods fit and predict

    @abstractmethod
    def inject(self, rules: Iterable[str]) -> None:
        pass


# TODO: find a better name. This class targets NN with constraining for classification using one-hot encoding.
class ConstrainingInjector(Injector):

    def __init__(self, predictor: Model, class_mapping: dict[str, int],
                 feature_mapping: dict[str, int], gamma: float = 1.):
        self.predictor = predictor
        self.class_mapping = class_mapping
        self.feature_mapping = feature_mapping
        self.gamma = gamma
        self._fuzzy_functions = None

    def inject(self, rules: Iterable[str]) -> None:
        visitor = Fuzzifier(self.class_mapping, self.feature_mapping)
        trees = [PrologParser(CommonTokenStream(PrologLexer(InputStream(rule)))).formula() for rule in rules]
        self._fuzzy_functions = [visitor.visit(tree) for tree in trees]
        predictor_output = self.predictor.layers[-1].output
        x = Concatenate(axis=1)([self.predictor.input, predictor_output])
        x = Lambda(self._cost, self.predictor.output.shape)(x)
        self.predictor = Model(self.predictor.input, x)

    def _cost(self, output_layer: Tensor) -> Tensor:
        input_len = self.predictor.input.shape[1]
        x, y = output_layer[:, :input_len], output_layer[:, input_len:]
        cost = stack([function(x, y) for function in self._fuzzy_functions], axis=1)
        return y + (cost / self.gamma)

    def remove(self) -> None:
        """
        Remove the constraining obtained by the injected rules.
        """
        self.predictor = Model(self.predictor.input, self.predictor.layers[-3].output)
