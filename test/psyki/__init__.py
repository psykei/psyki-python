from __future__ import annotations

from typing import Callable

from psyki.ski import Injector


def get_injector(choice: str) -> Callable:
    injectors = {'kill': Injector.kill,
                 'kins': Injector.kins,
                 'kbann': Injector.kbann}
    return injectors[choice]