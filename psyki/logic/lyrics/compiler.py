import tensorflow as tf
import abc
import numpy as np
from psyki.logic.lyrics import LOSS_MODE
from psyki.logic.lyrics.utils import cartesian
from psyki.logic.lyrics.world import World


class LogicTensor(tf.Tensor):
    def __init__(self, tensor, name=None):
        super(LogicTensor, self).__init__(
            op=tensor.op, value_index=tensor.value_index, dtype=tensor.dtype
        )
        self._name = name

    def __str__(self):
        a = self._name if self._name is not None else ""
        return a + super(LogicTensor, self).__str__()

    def __repr__(self):
        a = self._name if self._name is not None else ""
        return a + super(LogicTensor, self).__repr__()


class LogicVariableTensor(LogicTensor):
    def __init__(self, tensor, domain, name=None):
        super(LogicVariableTensor, self).__init__(tensor, name)
        self.id = World._var_id_provider.new_id()
        self.domain = domain

    def __hash__(self):
        return super.__hash__(self) + self.id


def variable(domain, name=None):
    domain = World.domains[domain] if isinstance(domain, str) else domain
    t = LogicVariableTensor(domain.tensor, domain, name)
    return t


class VariableDependentTensor(
    LogicTensor,
):
    """A variable-dependent tensor"""

    def __init__(self, tensor, variables):
        super(VariableDependentTensor, self).__init__(tensor)
        self.variables_list = self.create_variable_list(variables)

    def create_variable_list(self, args):
        return tuple(args)


class FormulaTensor(VariableDependentTensor):
    """A variable-dependent tensor which apply a function on its arguments"""

    def __init__(self, args):
        self.set_vars = None
        args = self.preprocess_args(args)
        t = self.apply_op(args)
        super(FormulaTensor, self).__init__(t, args)

    def create_variable_list(self, args):
        del args
        return tuple(sorted(self.set_vars, key=lambda x: x.id))

    def preprocess_args(self, args):
        self.set_vars = merge_vars(args)
        args = to_loss_by_g(args)
        args = to_coherent_shapes(args, self.set_vars)
        return args

    @abc.abstractmethod
    def apply_op(self, args):
        pass


class AtomTensor(FormulaTensor):
    def __init__(self, predicate, variables, tensor=None):
        self._tensor = tensor

        # Checking input variables compatibility with predicate domains
        assert len(predicate.domains) == len(variables)
        for i, v in enumerate(variables):
            assert (
                v.domain == predicate.domains[i]
                or predicate.domains[i] in v.domain.ancestors
            )

        self.predicate = predicate
        super(AtomTensor, self).__init__(variables)

    def apply_op(self, args):
        if self._tensor is None:
            cp = cartesian([v.domain.tensor for v in args])
            t = self.predicate.function(*cp)
            self._tensor = t
        tensor = tf.reshape(self._tensor, [v.domain.size for v in args])
        tensor = transposition(tensor, args)
        return tensor

    def preprocess_args(self, args):
        return args

    def create_variable_list(self, args):
        return tuple(sorted(args, key=lambda x: x.id))


def atom(predicate, variables):
    domains = tuple([v.domain for v in variables])
    key = (predicate, domains)
    if key not in World._predicates_cache:
        World._predicates_cache[key] = AtomTensor(predicate, variables)
        return World._predicates_cache[key]
    else:
        cached = World._predicates_cache[key]
        return AtomTensor(cached.predicate, variables, tensor=cached._tensor)


def to_loss_by_g(args):
    if World._evaluation_mode == LOSS_MODE:
        args = list(args)
        for i, a in enumerate(args):
            if isinstance(a, AtomTensor):
                args[i] = VariableDependentTensor(
                    tensor=World.generator(a), variables=a.variables_list
                )
        return tuple(args)
    else:
        return args


def transposition(t, args):
    temp = []
    for i, v in enumerate(args):
        temp.append(v.id)
    sorted_temp = sorted(temp)
    if sorted_temp == temp:
        return t
    else:
        map_ids = {}
        for i in sorted_temp:
            map_ids[i] = len(map_ids)
        transposition_order = []
        for i in temp:
            transposition_order.append(map_ids[i])
        return tf.transpose(t, transposition_order)


def to_coherent_shapes(args, set_vars):
    sorted_vars = sorted(set_vars, key=lambda v: v.id)
    args = list(args)
    for i, v in enumerate(sorted_vars):
        for j, arg in enumerate(args):
            if v not in arg.variables_list:
                temp = tf.expand_dims(arg, axis=i)
                if len(args) > 1:
                    multiplies = np.ones_like(temp.get_shape())
                    multiplies[i] = v.domain.size
                    temp = tf.tile(temp, multiplies)
                args[j] = VariableDependentTensor(
                    temp, tuple(set(args[j].variables_list).union([v]))
                )
    return tuple(args)


def merge_vars(args):
    return set(args[0].variables_list).union(*[a.variables_list for a in args[1:]])
