from psyki.logic.lyrics import LOSS_MODE
from psyki.logic.lyrics.utils import IDProvider


class World(object):
    # Map from domain name to domain object
    domains = {}

    # Map from function name to function object
    functions = {}

    # Map from relation name to relation object
    predicates = {}

    # Map from individual id to individual object
    individuals = {}

    # This is a storage for caching computational graphs
    _precomputed = {}

    _predicates_cache = {}

    _var_id_provider = IDProvider()

    generator = None

    tnorm = None

    _evaluation_mode = LOSS_MODE

    @staticmethod
    def reset():
        World.domains = {}
        World.functions = {}
        World.predicates = {}
        World.individuals = {}
        World._precomputed = {}
        World.generator = None
        World.tnorm = None
        World._predicates_cache = {}
        World._var_id_provider = IDProvider()
        World._evaluation_mode = LOSS_MODE
