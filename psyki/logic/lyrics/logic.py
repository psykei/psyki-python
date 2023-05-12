import tensorflow as tf
from psyki.logic.lyrics import *
from psyki.logic.lyrics.compiler import FormulaTensor
from psyki.logic.lyrics.world import World

EPSILON = 1e-12
FRANK = "FRANK"
SS = "SS"
PRODUCT = "PRODUCT"
ADAM = "Adam"
GD = "GD"


def r(x):
    return tf.where(x < EPSILON, x + EPSILON, x)


def log_b(x, b):
    numerator = tf.math.log(x + EPSILON)
    denominator = tf.math.log(tf.constant(b, dtype=numerator.dtype) + EPSILON)
    return numerator / denominator


class TNorm(object):
    def __init__(self):
        pass


class Generator(object):
    def __init__(self, p=None):
        pass


class ProductTNorm(TNorm):
    def __init__(self):
        super(ProductTNorm, self).__init__()

    def __call__(self, x, y):
        return x * y


class ProductGenerator(Generator):
    def __init__(self, p=None):
        super(ProductGenerator, self).__init__(p)

    def __call__(self, x):
        eps = 1e-7
        x = tf.clip_by_value(x, eps, 1.0 - eps)
        return -tf.math.log(x)


class FrankTNorm(TNorm):
    def __init__(self, p):
        super(FrankTNorm, self).__init__()
        self.p = p if p == "inf" else 1.0 * p

    def __call__(self, args):
        assert len(args) == 2, "N-ary t-norm not implemented for the Frank family"
        x = args[0]
        y = args[1]
        p = self.p
        if self.p == 0:
            return tf.minimum(x, y)
        elif self.p == 1:
            return x * y
        elif self.p == "inf":
            return tf.maximum(0.0, x + y - 1.0)

        else:
            return log_b(1.0 + (tf.pow(p, x) - 1) * (tf.pow(p, y) - 1) / (p - 1), p)


class FrankGenerator(Generator):
    def __init__(self, p=None):
        super(FrankGenerator, self).__init__(p)
        self.p = p

    def __call__(self, x):
        def r(x):
            return tf.minimum(x + 1e-6, 1)

        p = self.p

        res = tf.case(
            pred_fn_pairs=[
                (tf.equal(p, 1.0), lambda: -tf.math.log(x + 1e-6)),
                (tf.equal(p, float("inf")), lambda: 1 - x),
            ],
            default=lambda: tf.math.log((p - 1.0) / (tf.pow(p, r(x)) - 1)),
            exclusive=True,
        )
        return res


class SSTNorm(TNorm):
    def __init__(self, p=None):
        super(SSTNorm, self).__init__()
        self.p = p if p == "inf" or p == "-inf" else 1.0 * p

    def __call__(self, args):
        assert len(args) == 2, "N-ary t-norm not implemented for the SS family"
        x = args[0]
        y = args[1]

        def r(x):
            return tf.minimum(x + EPSILON, 1)

        p = self.p
        if self.p == "-inf":
            return tf.minimum(x, y)
        elif p < 0:
            return tf.pow(tf.pow(r(x), p) + tf.pow(r(y), p) - 1, 1 / p)
        elif self.p == 0:
            return x * y
        elif p > 0:
            return tf.pow(tf.maximum(0.0, tf.pow(r(x), p) + tf.pow(r(y), p) - 1), 1 / p)
        elif self.p == "inf":
            return tf.maximum(0.0, x + y - 1.0)


class SSGenerator(Generator):
    def __init__(self, p=None):
        super(SSGenerator, self).__init__(p)
        self.p = p

    def __call__(self, x):
        p = self.p
        return tf.cond(
            tf.equal(p, 0),
            lambda: -tf.math.log(x + EPSILON),
            lambda: (1 - tf.pow(x + EPSILON, p)) / p,
        )


def setTNorm(id, p=None):
    if id == FRANK:
        World.generator = FrankGenerator(p)
        World.tnorm = FrankTNorm(p)
    elif id == SS:
        World.generator = SSGenerator(p)
        World.tnorm = SSTNorm(p)
    elif id == PRODUCT:
        World.generator = ProductGenerator()
        World.tnorm = ProductTNorm()
    else:
        raise Exception("Unknown id for TNorm and Generator Family")


# -------- CONNECTIVE TENSORS ----------#


class AndNG(FormulaTensor):
    def apply_op(self, args):
        return tf.add_n(args)


class AndNT(FormulaTensor):
    def __init__(self, args):
        self.tnorm = World.tnorm
        super(AndNT, self).__init__(args)

    def apply_op(self, args):
        assert len(args) == 2
        return self.tnorm(args)


class OrNG(FormulaTensor):
    def apply_op(self, args):
        t = tf.stack(args, axis=1)
        return tf.reduce_min(t)


class OrNT(FormulaTensor):
    def __init__(self, args):
        self.tnorm = World.tnorm
        super(OrNT, self).__init__(args)

    def apply_op(self, args):
        t = tf.stack(args, axis=1)
        return tf.reduce_max(t)


class ImpliesG(FormulaTensor):
    def apply_op(self, args):
        assert len(args) == 2
        x = args[0]
        y = args[1]
        return tf.where(x < y, y - x, tf.zeros_like(x))


class ImpliesT(FormulaTensor):
    def apply_op(self, args):
        assert len(args) == 2
        x = args[0]
        y = args[1]
        return tf.maximum(1 - x, y)


class IffG(FormulaTensor):
    def apply_op(self, args):
        assert len(args) == 2
        x = args[0]
        y = args[1]
        return tf.abs(x - y)


class IffT(FormulaTensor):
    def apply_op(self, args):
        assert len(args) == 2
        x = args[0]
        y = args[1]
        return self.tnorm([tf.maximum(1 - x, y), tf.maximum(x, 1 - y)])


class Quantifier(object):
    def __init__(self, variable, formula):
        self.var_index = formula.variables_list.index(variable)


class ForAllG(FormulaTensor, Quantifier):
    def __init__(self, variable, formula):
        self.variable = variable
        Quantifier.__init__(self, variable, formula)
        FormulaTensor.__init__(self, (formula,))

    def apply_op(self, args):
        return tf.reduce_sum(args[0], self.var_index)


class ForAllT(FormulaTensor, Quantifier):
    def __init__(self, variable, formula):
        self.variable = variable
        Quantifier.__init__(self, variable, formula)
        FormulaTensor.__init__(self, (formula,))

    def apply_op(self, args):
        print(
            "LOGIC Evaluation do not support Quantifiers. Weak Quantifiers are exploited"
        )
        return tf.reduce_min(args[0], axis=self.var_index)


class LogicFactory:
    _loss_map = {AND: AndNG, OR: OrNG, IMPLIES: ImpliesG, FORALL: ForAllG, IFF: IffG}
    _logic_map = {AND: AndNT, OR: OrNT, IMPLIES: ImpliesT, FORALL: ForAllT, IFF: IffT}

    @staticmethod
    def create(operator):
        if World._evaluation_mode == LOSS_MODE:
            return LogicFactory._loss_map[operator]
        else:
            return LogicFactory._logic_map[operator]


# ------------ PROXIES OPERATIONS --------------------#
def and_n(*args):
    res = LogicFactory.create(AND)(args)
    return res


def or_n(*args):
    return LogicFactory.create(OR)(args)


def implies(*args):
    return LogicFactory.create(IMPLIES)(args)


def forall(variable, formula):
    return LogicFactory.create(FORALL)(variable, formula)


def exists(variable, formula):
    return LogicFactory.create(EXISTS)(variable, formula)


def iff(*args):
    return LogicFactory.create(IFF)(args)


class Knowledge:
    def __init__(self):
        self.rules = []

    def add(self, rule, weight=None):
        if weight is not None:
            self.rules.append(weight * rule)
        else:
            self.rules.append(rule)

    def loss(self):
        return tf.add_n(self.rules)
