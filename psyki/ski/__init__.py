from abc import ABC, abstractmethod
from typing import Any, Callable, Iterable
from tensorflow import maximum, minimum, stack, Tensor, reduce_max, SparseTensor, cast, tile, reshape, shape, constant
from tensorflow.keras.layers import Concatenate, Lambda, Dense, Dot
from tensorflow.keras import Model
from tensorflow.keras.backend import to_dense
from tensorflow.python.ops.array_ops import gather
from tensorflow.keras.layers import Minimum, Maximum
from tensorflow.python.ops.init_ops import Zeros, constant_initializer, Ones
from psyki.utils import eta, eta_one_abs
from resources.dist.resources.PrologParser import PrologParser
from resources.dist.resources.PrologVisitor import PrologVisitor


class Fuzzifier(PrologVisitor):

    def __init__(self, class_mapping: dict[str, int], feature_mapping: dict[str, int]):
        self.feature_mapping = feature_mapping
        self.class_mapping = {string: cast(to_dense(SparseTensor([[0, index]], [1.], (1, len(class_mapping)))), float)
                              for string, index in class_mapping.items()}

    def clause_expression(self, ctx: PrologParser.ClauseExpressionContext or PrologParser.ClauseExpressionNoParContext):
        l, r = self.visit(ctx.left), self.visit(ctx.right)
        operation = {
            '∧': lambda x: eta(maximum(l(x), r(x))),
            '∨': lambda x: eta(minimum(l(x), r(x))),
            '→': lambda x: eta(l(x) - r(x)),
            '↔': lambda x: eta(abs(l(x) - r(x))),
            '=': lambda x: eta(abs(l(x) - r(x))),
            '<': lambda x: eta(constant(1.) - eta(constant(1.) - eta(l(x) - r(x)))),
            '≤': lambda x: eta(constant(1.) - eta(maximum(eta(constant(1.) - eta(l(x) - r(x))),
                                                          eta(constant(1.) - eta(abs(l(x) - r(x))))))),
            '>': lambda x: eta(maximum(eta(constant(1.) - eta(l(x) - r(x))),
                                       eta(constant(1.) - eta(abs(l(x) - r(x)))))),
            '≥': lambda x: eta(constant(1.) - eta(l(x) - r(x))),
            '+': lambda x: l(x) + r(x),
            '*': lambda x: l(x) * r(x)
        }
        return operation.get(ctx.op.text)

    # Visit a parse tree produced by folParser#formula.
    def visitFormula(self, ctx: PrologParser.FormulaContext):
        l = lambda y: eta(reduce_max(abs(tile(self.visit(ctx.args), (shape(y)[0], 1)) - y), axis=1))
        r = self.visit(ctx.right)
        return lambda x, y: eta(l(y) - r(x))

    # Visit a parse tree produced by PrologParser#ClauseExpressionNoPar.
    def visitClauseExpressionNoPar(self, ctx: PrologParser.ClauseExpressionNoParContext):
        return self.clause_expression(ctx)

    # Visit a parse tree produced by folParser#ClauseExpression.
    def visitClauseExpression(self, ctx: PrologParser.ClauseExpressionContext):
        return self.clause_expression(ctx)

    # Visit a parse tree produced by PrologParser#ConstFunctor.
    def visitConstFunctor(self, ctx: PrologParser.ConstFunctorContext):
        return reshape(self.class_mapping[ctx.fun.text], (1, len(self.class_mapping)))

    # Visit a parse tree produced by PrologParser#ConstNumber.
    def visitConstNumber(self, ctx: PrologParser.ConstNumberContext):
        return lambda _: float(ctx.num.text)

    # Visit a parse tree produced by folParser#TermVar.
    def visitTermVar(self, ctx: PrologParser.TermVarContext):
        var = ctx.var.text
        return lambda x: x[:, self.feature_mapping[var]] \
            if var in self.feature_mapping.keys() else self.visitChildren(ctx)


class SubNetworkBuilder(PrologVisitor):

    def __init__(self, predictor_input: Tensor, feature_mapping: dict[str, int]):
        self.predictor_input = predictor_input
        self.feature_mapping = feature_mapping

    def clause_expression(self, ctx: PrologParser.ClauseExpressionContext or PrologParser.ClauseExpressionNoParContext):
        previous_layer = [self.visit(ctx.left), self.visit(ctx.right)]
        operation = {
            '∧': Minimum()(previous_layer),
            '∨': Maximum()(previous_layer),
            '→': None,
            '↔': None,
            '=': Dense(1, kernel_initializer=constant_initializer([1, -1]),
                       activation=eta_one_abs, trainable=False)(Concatenate(axis=1)(previous_layer)),
            '<': Dense(1, kernel_initializer=constant_initializer([-1, 1]),
                       activation=eta, trainable=False)(Concatenate(axis=1)(previous_layer)),
            '≤': Maximum()([Dense(1, kernel_initializer=constant_initializer([-1, 1]),
                                  activation=eta, trainable=False)(Concatenate(axis=1)(previous_layer)),
                            Dense(1, kernel_initializer=constant_initializer([1, -1]),
                                  activation=eta_one_abs, trainable=False)(Concatenate(axis=1)(previous_layer))]),
            '>': Dense(1, kernel_initializer=constant_initializer([1, -1]),
                       activation=eta, trainable=False)(Concatenate(axis=1)(previous_layer)),
            '≥': Maximum()([Dense(1, kernel_initializer=constant_initializer([1, -1]),
                                  activation=eta, trainable=False)(Concatenate(axis=1)(previous_layer)),
                            Dense(1, kernel_initializer=constant_initializer([1, -1]),
                                  activation=eta_one_abs, trainable=False)(Concatenate(axis=1)(previous_layer))]),
            '+': Dense(1, kernel_initializer=Ones, activation='linear', trainable=False)
            (Concatenate(axis=1)(previous_layer)),
            '*': Dot(axes=1)(previous_layer)
        }
        return operation.get(ctx.op.text)

    # Visit a parse tree produced by PrologParser#ClauseExpressionNoPar.
    def visitClauseExpressionNoPar(self, ctx: PrologParser.ClauseExpressionNoParContext):
        return self.clause_expression(ctx)

    # Visit a parse tree produced by folParser#ClauseExpression.
    def visitClauseExpression(self, ctx: PrologParser.ClauseExpressionContext):
        return self.clause_expression(ctx)

    # Visit a parse tree produced by PrologParser#ConstNumber.
    def visitConstNumber(self, ctx: PrologParser.ConstNumberContext):
        return Dense(1, kernel_initializer=Zeros, bias_initializer=constant_initializer(float(ctx.num.text)),
                     trainable=False, activation='linear')(self.predictor_input)

    # Visit a parse tree produced by folParser#TermVar.
    def visitTermVar(self, ctx: PrologParser.TermVarContext):
        var = ctx.var.text
        return Lambda(lambda x: gather(x, [self.feature_mapping[var]], axis=1))(self.predictor_input)


class Injector(ABC):
    """
    An injector is a class that allows a sub-symbolic predictor to exploit prior symbolic knowledge.
    The knowledge is provided via rules in some sort of logic form (e.g. FOL, Skolem, Horn).
    """
    predictor: Any  # Any class that has methods fit and predict

    @abstractmethod
    def inject(self, rules: dict[str, PrologParser]) -> None:
        pass


# TODO: find a better name. This class targets NN with constraining for classification using one-hot encoding.
class ConstrainingInjector(Injector):

    def __init__(self, predictor: Model, class_mapping: dict[str, int],
                 feature_mapping: dict[str, int], gamma: float = 1.):
        self.predictor: Model = predictor
        self.class_mapping: dict[str, int] = class_mapping
        self.feature_mapping: dict[str, int] = feature_mapping
        self.gamma: float = gamma
        self._fuzzy_functions: Iterable[Callable] = ()

    def inject(self, rules: dict[str, PrologParser]) -> None:
        visitor = Fuzzifier(self.class_mapping, self.feature_mapping)
        self._fuzzy_functions = [visitor.visit(tree.formula()) for tree in rules.values()]
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


# TODO: find a better name. This class targets NN with structuring using one-hot encoding.
class StructuringInjector(Injector):

    def __init__(self, predictor: Model, feature_mapping: dict[str, int]):
        self.predictor: Model = predictor
        self.feature_mapping: dict[str, int] = feature_mapping

    def inject(self, rules: dict[str, PrologParser]) -> None:
        visitor = SubNetworkBuilder(self.predictor.input, self.feature_mapping)
        predictor_output: Tensor = self.predictor.layers[-2].output
        modules = [visitor.visit(tree.formula()) for tree in rules.values()]
        neurons: int = self.predictor.layers[-1].output.shape[1]
        activation: Callable = self.predictor.layers[-1].activation
        new_predictor = Dense(neurons, activation=activation)(Concatenate(axis=1)([predictor_output] + modules))
        self.predictor = Model(self.predictor.input, new_predictor)
