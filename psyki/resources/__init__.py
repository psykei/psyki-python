import re
from os import popen, system
from pathlib import Path
from psyki import PSYKI_PATH

PATH = Path(__file__).parents[0]


def create_antlr4_parser(file: str, path: str):
    antlr4_version = re.split(r'=', popen('cat ' + str(PSYKI_PATH / 'requirements.txt') + ' | grep antlr4').read())[1][:-1]
    system('wget https://www.antlr.org/download/antlr-' + antlr4_version + '-complete.jar -O' + str(PSYKI_PATH / ('antlr-' + antlr4_version + '-complete.jar')))
    system('export CLASSPATH="' + str(PSYKI_PATH / ('antlr-' + antlr4_version + '-complete.jar:$CLASSPATH"')))
    system('java -jar ' + str(PSYKI_PATH / ('antlr-' + antlr4_version + '-complete.jar -Dlanguage=Python3 ' + file + ' -visitor -o ' + path)))
    system('rm ' + str(PSYKI_PATH / ('antlr-' + antlr4_version + '-complete.jar')))
