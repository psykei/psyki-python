import pandas as pd
from test.resources.data.pipelines.adult_data_pipeline import AdultLoader


def load_dataset(dataset_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load a dataset.

    :param dataset_name: name of the dataset
    :return: train and test dataframes
    """

    if dataset_name == "adult":
        train, test = AdultLoader().load_preprocessed_split(
            one_hot=False, validation=False
        )
    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")

    # Check of there are nan values in the dataset
    if train.isnull().values.any() or test.isnull().values.any():
        raise ValueError(f"Dataset {dataset_name} contains nan values.")

    return train, test
