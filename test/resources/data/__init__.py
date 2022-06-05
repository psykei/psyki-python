import re
from pathlib import Path
from typing import Any, Iterable
import numpy as np
import pandas as pd
from test.resources.data.poker import PATH as POKER_PATH
from test.resources.data.splice_junction import PATH as SJ_PATH


PATH = Path(__file__).parents[0]
DATA_REGISTER = {
    "poker": POKER_PATH,
    "splice_junction": SJ_PATH
}


def get_dataset(dataset_domain: str = "poker", dataset_name: str = "data") -> Any:
    return np.genfromtxt(str(DATA_REGISTER[dataset_domain] / dataset_name) + '.csv', delimiter=',', dtype=float)


def get_dataset_dataframe(dataset_domain: str = "poker", dataset_name: str = "data") -> pd.DataFrame:
    return _get_data(str(DATA_REGISTER[dataset_domain] / dataset_name) + '.txt')


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
