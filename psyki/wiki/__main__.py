import os
from typing import Callable
from psyki.ski import PATH as INJECTORS_PATH
from psyki.logic import PATH as FUZZIFIERS_PATH
from psyki.utils import execute_command


def commands() -> dict[str, Callable]:
    return {
        'list': elicit,
    }


def elicit(what: str = 'injectors'):
    mapping_paths = {
        'injectors': INJECTORS_PATH,
        'fuzzifiers': FUZZIFIERS_PATH,
    }
    path = mapping_paths[what]
    what_dirs_str = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name)) and name[0] != '_']
    what_dirs = [path / what_dir for what_dir in what_dirs_str]
    if what == 'fuzzifiers':
        for i, what_dir in enumerate(what_dirs):
            what_dir = what_dir / what
            if os.path.exists(what_dir):
                what_sub_dirs = [name for name in os.listdir(what_dir) if os.path.isdir(os.path.join(what_dir, name)) and name[0] != '_']
                what_dirs_str[i] = what_dirs_str[i] + '\n   |- ' + '\n   |- '.join(what_sub_dirs)
    print('\nAvailable ' + what + ' are:\n - ' + '\n - '.join(what_dirs_str))


if __name__ == '__main__':
    execute_command(commands)
