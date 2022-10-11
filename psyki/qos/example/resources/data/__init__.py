from pathlib import Path
from typing import Any
import numpy as np


PATH = Path(__file__).parents[0]


def get_dataset(dataset_name: str = "train") -> Any:
    return np.genfromtxt(str(PATH / dataset_name) + '.csv', delimiter=',', dtype=float)
