from .world import World
from .compiler import AtomTensor


class Predicate(object):
    def __init__(self, label, domains, function):
        if label in World.predicates:
            raise Exception("Predicate %s already exists" % label)
        self.label = label
        self.domains = [World.domains[d] if isinstance(d, str) else d for d in domains]
        World.predicates[label] = self
        self.arity = len(self.domains)
        if function is None:
            raise NotImplementedError(
                "Default function implementation in Relation not yet implemented"
            )
        else:
            self.function = function

    # TODO Experimental Feature
    def __call__(self, *variables):
        domains = tuple([v.domain for v in variables])
        key = (self, domains)
        if key not in World._predicates_cache:
            World._predicates_cache[key] = AtomTensor(self, variables)
            return World._predicates_cache[key]
        else:
            cached = World._predicates_cache[key]
            return AtomTensor(cached.predicate, variables, tensor=cached._tensor)
