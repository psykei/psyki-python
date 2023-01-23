import re
from pathlib import Path
from typing import Iterable
import pandas as pd

PATH = Path(__file__).parents[0]


class SpliceJunction(object):
    base_features = ['a', 'c', 'g', 't']
    aggregate_feature_mapping = {'a': ('a',),
                                 'c': ('c',),
                                 'g': ('g',),
                                 't': ('t',),
                                 'd': ('a', 'g', 't'),
                                 'm': ('a', 'c'),
                                 'n': ('a', 'c', 'g', 't'),
                                 'r': ('a', 'g'),
                                 's': ('c', 'g'),
                                 'y': ('c', 't')}
    indices = list(range(-30, 0)) + list(range(1, 30)),
    class_mapping = {'ei': 0,
                     'ie': 1,
                     'n': 2}
    feature_mapping = {'X' + ('_' if j < 0 else '') + str(abs(j)) + f: k + i * 4 for i, j in
                       enumerate(list(range(-30, 0)) + list(range(1, 31))) for k, f in enumerate(['a', 'c', 'g', 't'])}


class Poker(object):
    feature_mapping = {'S1': 0,
                       'R1': 1,
                       'S2': 2,
                       'R2': 3,
                       'S3': 4,
                       'R3': 5,
                       'S4': 6,
                       'R4': 7,
                       'S5': 8,
                       'R5': 9}
    class_mapping = {'nothing': 0,
                     'pair': 1,
                     'two': 2,
                     'three': 3,
                     'straight': 4,
                     'flush': 5,
                     'full': 6,
                     'four': 7,
                     'straight_flush': 8,
                     'royal_flush': 9}


class Iris(object):
    class_mapping = {'setosa': 0, 'virginica': 1, 'versicolor': 2}
    feature_mapping = {'SepalLength': 0, 'SepalWidth': 1, 'PetalLength': 2, 'PetalWidth': 3}


def get_dataset(filename: str) -> pd.DataFrame:
    return pd.read_csv(str(PATH / filename))


def get_splice_junction_processed_dataset(filename: str) -> pd.DataFrame:
    x = []
    with open(str(PATH / filename), encoding="utf8") as f:
        for row in f:
            row = re.sub('\n', '', row)
            label, _, features = row.split(',')
            features = list(f for f in features.lower())
            features.append(label.lower())
            x.append(features)
    data = pd.DataFrame(x)
    y = _data_to_int(data.iloc[:, -1:], SpliceJunction.class_mapping)
    x = _get_binary_data(data.iloc[:, :-1], SpliceJunction.aggregate_feature_mapping)
    y.columns = [x.shape[1]]
    return x.join(y)


def _data_to_int(data: pd.DataFrame, mapping: dict[str: int]) -> pd.DataFrame:
    return data.applymap(lambda x: mapping[x] if x in mapping.keys() else x)


def _get_binary_data(data: pd.DataFrame, mapping: dict[str: set[str]]) -> pd.DataFrame:
    sub_features = sorted(_get_values(mapping))
    results = []
    for _, row in data.iterrows():
        row_result = []
        for value in row:
            positive_features = mapping[value]
            for feature in sub_features:
                row_result.append(1 if feature in positive_features else 0)
        results.append(row_result)
    return pd.DataFrame(results, dtype=int)


def _get_values(mapping: dict[str: set[str]]) -> Iterable[str]:
    result = set()
    for values_set in mapping.values():
        for value in values_set:
            result.add(value)
    return result
