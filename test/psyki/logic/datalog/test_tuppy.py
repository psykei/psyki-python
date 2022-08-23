import unittest
from psyki.logic.datalog.grammar.adapters.antlr4 import to_prolog_string
from psyki.logic.datalog.grammar.adapters.tuppy import prolog_to_datalog
from test.resources.rules.iris import PATH
from psyki.logic.prolog.grammar.adapters.tuppy import file_to_prolog


class TestTuppy(unittest.TestCase):

    iris_kb = 'kb-prolog.txt'

    def test_from_text_to_formula(self):
        prolog_theory = file_to_prolog(PATH / self.iris_kb)
        datalog_formula = prolog_to_datalog(prolog_theory)

if __name__ == '__main__':
    unittest.main()
