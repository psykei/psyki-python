from pathlib import Path

PATH = Path(__file__).parents[0]


def get_indices() -> list[int]:
    return list(range(-30, 0)) + list(range(1, 31))


FEATURES = ['a', 'c', 'g', 't']

CLASS_MAPPING = {'ei': 0,
                 'ie': 1,
                 'n': 2}

AGGREGATE_FEATURE_MAPPING = {'a': ('a',),
                             'c': ('c',),
                             'g': ('g',),
                             't': ('t',),
                             'd': ('a', 'g', 't'),
                             'm': ('a', 'c'),
                             'n': ('a', 'c', 'g', 't'),
                             'r': ('a', 'g'),
                             's': ('c', 'g'),
                             'y': ('c', 't')}

SPLICE_JUNCTION_INDICES = get_indices()


def get_splice_junction_feature_mapping() -> dict[str: int]:
    return _get_feature_mapping(SPLICE_JUNCTION_INDICES)


def get_splice_junction_extended_feature_mapping() -> dict[str: int]:
    return _get_extended_feature_mapping(FEATURES, SPLICE_JUNCTION_INDICES)


def _get_feature_mapping(variable_indices: list[int]) -> dict[str: int]:
    return {'X' + ('_' if j < 0 else '') + str(abs(j)): i for i, j in enumerate(variable_indices)}


def _get_extended_feature_mapping(features: list[str], variable_indices: list[int]) -> dict[str: int]:
    result = {'X' + ('_' if j < 0 else '') + str(abs(j)) + f: k + i * len(features)
            for i, j in enumerate(variable_indices) for k, f in enumerate(features)}
    return result
