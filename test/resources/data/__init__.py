from __future__ import annotations
import math
import re
from abc import ABCMeta
from pathlib import Path
from typing import Iterable
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder
from psyki.logic import Formula, Theory
from psyki.logic.prolog import TuProlog
from test.resources.knowledge import PATH as KNOWLEDGE_PATH

PATH = Path(__file__).parents[0]
UCI_URL: str = "https://archive.ics.uci.edu/ml/machine-learning-databases/"


class Dataset(ABCMeta):
    name: str = "Dataset name"
    filename: str = "Dataset filename"
    test_filename: str = "Dataset test filename"
    knowledge_filename: str = "Dataset knowledge filename"
    data_url: str = None
    data_url_test: str = None
    class_mapping: dict[str, int] = {}
    features: list[str] = []
    target: list[str] = []
    preprocess: bool = False
    need_download: bool = True

    @classmethod
    @property
    def is_downloaded(mcs) -> bool:
        return (PATH / mcs.filename).is_file()

    @classmethod
    @property
    def is_test_downloaded(mcs) -> bool:
        return (
            (PATH / mcs.test_filename).is_file()
            if mcs.test_filename is not None
            else False
        )

    @classmethod
    def download(mcs) -> None:
        if mcs.need_download and not mcs.is_downloaded:
            d: pd.DataFrame = pd.read_csv(
                mcs.data_url, sep=r",\s*", header=None, encoding="utf8"
            )
            d.to_csv(PATH / mcs.filename, index=False, header=False)
        if mcs.data_url_test is not None and not mcs.is_test_downloaded:
            d: pd.DataFrame = pd.read_csv(
                mcs.data_url_test, sep=r",\s*", header=None, encoding="utf8"
            )
            d.to_csv(PATH / mcs.test_filename, index=False, header=False)

    @classmethod
    def get_knowledge(mcs) -> list[Formula]:
        return TuProlog.from_file(str(KNOWLEDGE_PATH / mcs.knowledge_filename))

    @classmethod
    def get_theory(mcs) -> Theory:
        return Theory(mcs.get_knowledge(), mcs.get_train(), mcs.class_mapping)

    @classmethod
    def get_train(mcs) -> pd.DataFrame:
        if mcs.preprocess:
            return mcs.get_processed_dataset(mcs.filename)
        else:
            return pd.read_csv(PATH / mcs.filename)

    @classmethod
    def get_test(mcs) -> pd.DataFrame:
        if mcs.preprocess:
            return mcs.get_processed_dataset(mcs.test_filename)
        else:
            return pd.read_csv(PATH / mcs.filename)

    @staticmethod
    def get_processed_dataset(filename: str) -> pd.DataFrame:
        pass


class SpliceJunction(Dataset):
    name = "Splice Junction"
    filename = "splice-junction.csv"
    knowledge_filename = str(KNOWLEDGE_PATH / "splice-junction.pl")
    data_url = UCI_URL + "molecular-biology/splice-junction-gene-sequences/splice.data"
    data_url_test = None
    class_mapping = {"ei": 0, "ie": 1, "n": 2}
    features = {
        "X" + ("_" if j < 0 else "") + str(abs(j)) + f: k + i * 4
        for i, j in enumerate(list(range(-30, 0)) + list(range(1, 31)))
        for k, f in enumerate(["a", "c", "g", "t"])
    }
    preprocess = True

    @staticmethod
    def get_processed_dataset(filename: str) -> pd.DataFrame:
        def _data_to_int(d: pd.DataFrame, mapping: dict[str:int]) -> pd.DataFrame:
            return d.applymap(lambda x: mapping[x] if x in mapping.keys() else x)

        def _get_values(mapping: dict[str : set[str]]) -> Iterable[str]:
            result = set()
            for values_set in mapping.values():
                for value in values_set:
                    result.add(value)
            return result

        def _get_binary_data(
            d: pd.DataFrame, mapping: dict[str : set[str]]
        ) -> pd.DataFrame:
            sub_features = sorted(_get_values(mapping))
            results = []
            for _, row in d.iterrows():
                row_result = []
                for value in row:
                    positive_features = mapping[value]
                    for feature in sub_features:
                        row_result.append(1 if feature in positive_features else 0)
                results.append(row_result)
            return pd.DataFrame(results, dtype=int)

        aggregate_mapping = {
            "a": ("a",),
            "c": ("c",),
            "g": ("g",),
            "t": ("t",),
            "d": ("a", "g", "t"),
            "m": ("a", "c"),
            "n": ("a", "c", "g", "t"),
            "r": ("a", "g"),
            "s": ("c", "g"),
            "y": ("c", "t"),
        }
        x = []
        with open(str(PATH / filename), encoding="utf8") as f:
            for row in f:
                row = re.sub("\n", "", row)
                label, _, features = row.split(",")
                features = list(f for f in features.lower())
                features.append(label.lower())
                x.append(features)
        data = pd.DataFrame(x)
        y = _data_to_int(data.iloc[:, -1:], SpliceJunction.class_mapping)
        x = _get_binary_data(data.iloc[:, :-1], aggregate_mapping)
        y.columns = [x.shape[1]]
        x.columns = SpliceJunction.features.keys()
        return x.join(y)


class Poker(Dataset):
    name = "Poker"
    filename = "poker-train.csv"
    test_filename = "poker-test.csv"
    knowledge_filename = str(KNOWLEDGE_PATH / "poker.pl")
    data_url = UCI_URL + "poker/poker-hand-training-true.data"
    data_url_test = UCI_URL + "poker/poker-hand-testing.data"
    features = [f"{k}{math.ceil((i + 1) / 2)}" for i, k in enumerate(5 * ["S", "R"])]
    class_mapping = {
        k: v
        for v, k in enumerate(
            [
                "nothing",
                "pair",
                "two",
                "three",
                "straight",
                "flush",
                "full",
                "four",
                "straight_flush",
                "royal_flush",
            ]
        )
    }
    preprocess = True

    @staticmethod
    def get_processed_dataset(filename: str) -> pd.DataFrame:
        data = pd.read_csv(PATH / filename)
        data.columns = Poker.features + ["class"]
        return data


class Iris(Dataset):
    name = "Iris"
    filename = None
    test_filename = None
    knowledge_filename = str(KNOWLEDGE_PATH / "iris.pl")
    class_mapping = {"setosa": 0, "virginica": 1, "versicolor": 2}
    features = ["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]
    need_download = False

    @classmethod
    def get_train(mcs) -> pd.DataFrame:
        x, y = load_iris(return_X_y=True, as_frame=True)
        encoder = OneHotEncoder(sparse=False)
        encoder.fit_transform([y])
        x.columns = list(Iris.features)
        return x.join(y)

    @classmethod
    def get_test(mcs) -> pd.DataFrame:
        return Iris.get_train()


DATASETS = [SpliceJunction, Poker, Iris]
