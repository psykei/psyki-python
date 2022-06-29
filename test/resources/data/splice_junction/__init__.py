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
