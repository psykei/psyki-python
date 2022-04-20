import re
from os import popen, system
from pathlib import Path

PATH = Path(__file__).parents[0]


def create_antlr4_parser(file: str, path: str):
    antlr4_version = re.split(r'=', popen('cat requirements.txt | grep antlr4').read())[1][:-1]
    system('wget https://www.antlr.org/download/antlr-' + antlr4_version + '-complete.jar')
    system('export CLASSPATH="./antlr-' + antlr4_version + '-complete.jar:$CLASSPATH"')
    system('java -jar ./antlr-' + antlr4_version + '-complete.jar -Dlanguage=Python3 ' + file + ' -visitor -o ' + path)
    system('rm ./antlr-' + antlr4_version + '-complete.jar')
