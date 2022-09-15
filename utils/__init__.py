import sys
from typing import Callable


def execute_command(commands: Callable):
    if len(sys.argv) > 1:
        first_arg = sys.argv[1]
        other_arguments = sys.argv[2:] if len(sys.argv) > 2 else []
        command = commands()[first_arg]
        command(*other_arguments)


def initialize_antlr4(file: str):
    import re
    from os import system, popen
    antlr4_version = re.split(r'=', popen('cat requirements.txt | grep antlr4').read())[1][:-1]
    system('wget https://www.antlr.org/download/antlr-' + antlr4_version + '-complete.jar')
    system('export CLASSPATH="./antlr-' + antlr4_version + '-complete.jar:$CLASSPATH"')
    system(
        'java -jar ./antlr-' + antlr4_version + '-complete.jar -Dlanguage=Python3 ' + file + ' -visitor -o psyki/resources/dist')
    system('rm ./antlr-' + antlr4_version + '-complete.jar')