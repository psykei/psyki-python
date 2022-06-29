import re
from pathlib import Path
from typing import Any, Iterable
import numpy as np
import pandas as pd
from test.resources.data.poker import PATH as POKER_PATH
from test.resources.data.splice_junction import PATH as SJ_PATH, get_indices as get_spice_junction_indices, \
    CLASS_MAPPING, AGGREGATE_FEATURE_MAPPING, FEATURES

PATH = Path(__file__).parents[0]
SPLICE_JUNCTION_INDICES = get_spice_junction_indices()


DATA_REGISTER = {
    "poker": POKER_PATH,
    "splice_junction": SJ_PATH
}


def get_dataset(dataset_domain: str = "poker", dataset_name: str = "data") -> Any:
    return np.genfromtxt(str(DATA_REGISTER[dataset_domain] / dataset_name) + '.csv', delimiter=',', dtype=float)


def get_dataset_dataframe(dataset_domain: str = "poker", dataset_name: str = "data") -> pd.DataFrame:
    return _get_data(str(DATA_REGISTER[dataset_domain] / dataset_name) + '.txt')


def get_splice_junction_processed_dataset(filename: str) -> pd.DataFrame:
    data = get_splice_junction_data(filename)
    y = data_to_int(data.iloc[:, -1:], CLASS_MAPPING)
    x = get_binary_data(data.iloc[:, :-1], AGGREGATE_FEATURE_MAPPING)
    y.columns = [x.shape[1]]
    return x.join(y)


def get_splice_junction_data(filename: str) -> pd.DataFrame:
    return _get_data(str(SJ_PATH / filename) + '.txt')


def data_to_int(data: pd.DataFrame, mapping: dict[str: int]) -> pd.DataFrame:
    return data.applymap(lambda x: mapping[x] if x in mapping.keys() else x)


def _get_data(file: str) -> pd.DataFrame:
    x = []
    with open(file) as f:
        for row in f:
            row = re.sub('\n', '', row)
            label, _, features = row.split(',')
            features = list(f for f in features.lower())
            features.append(label.lower())
            x.append(features)
    return pd.DataFrame(x)


def get_binary_data(data: pd.DataFrame, mapping: dict[str: set[str]]) -> pd.DataFrame:
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


def get_splice_junction_feature_mapping(variable_indices: list[int] = SPLICE_JUNCTION_INDICES) -> dict[str: int]:
    return _get_feature_mapping(variable_indices)


def get_splice_junction_extended_feature_mapping(features: list[str] = FEATURES,
                                                 variable_indices: list[int] = SPLICE_JUNCTION_INDICES
                                                 ) -> dict[str: int]:
    return _get_extended_feature_mapping(features, variable_indices)


def _get_feature_mapping(variable_indices: list[int]) -> dict[str: int]:
    return {'X' + ('_' if j < 0 else '') + str(abs(j)): i for i, j in enumerate(variable_indices)}


def _get_extended_feature_mapping(features: list[str], variable_indices: list[int]) -> dict[str: int]:
    result = {'X' + ('_' if j < 0 else '') + str(abs(j)) + f: k + i * len(features)
            for i, j in enumerate(variable_indices) for k, f in enumerate(features)}
    return result
