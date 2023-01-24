from __future__ import annotations
import pandas as pd
from sklearn.model_selection import train_test_split


def split_dataset(dataset) -> dict[str: pd.DataFrame]:
    # Split dataset into train and test
    train, test = train_test_split(dataset, test_size=0.3, random_state=0)
    train_x, train_y = train.iloc[:, :-1], train.iloc[:, -1]
    test_x, test_y = test.iloc[:, :-1], test.iloc[:, -1]
    return {"train_x": train_x, "train_y": train_y, "test_x": test_x, "test_y": test_y}
