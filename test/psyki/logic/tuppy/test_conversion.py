import unittest
from psyki.logic.datalog.grammar.adapters.tuppy import prolog_to_datalog
from test.resources.rules.iris import PATH
from psyki.logic.prolog.grammar.adapters.tuppy import file_to_prolog


class TestConversion(unittest.TestCase):

    iris_kb = 'kb-prolog.txt'

    def test_prolog_from_text_to_datalog(self):
        common_head = 'iris(PetalLength,PetalWidth,SepalLength,SepalWidth'
        expected_result = [common_head + ',virginica) <- not(PetalWidth >= 0.664341, PetalWidth < 1.651423)',
                           common_head + ',setosa) <- PetalWidth =< 1.651423',
                           common_head + ',versicolor) <- True']
        prolog_theory = file_to_prolog(PATH / self.iris_kb)
        datalog_formulae = prolog_to_datalog(prolog_theory)
        for i, formula in enumerate(datalog_formulae):
            self.assertEqual(expected_result[i], str(formula))


if __name__ == '__main__':
    unittest.main()
