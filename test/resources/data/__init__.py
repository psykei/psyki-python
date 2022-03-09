from pathlib import Path
from typing import Any
import numpy as np

PATH = Path(__file__).parents[0]


def get_dataset(name: str) -> Any:
    return np.genfromtxt(str(PATH / name) + '.csv', delimiter=',', dtype=float)
