import unittest
from test.resources.knowledge import PATH as KNOWLEDGE_PATH
from psyki.logic.prolog import TuProlog


class TestFormulaOptimization(unittest.TestCase):
    knowledge = TuProlog.from_file(KNOWLEDGE_PATH / 'splice-junction.pl').formulae

    def test_conjunction(self):
        first_formula = self.knowledge[0]
        print(first_formula)
        first_formula.optimize()
        self.assertTrue(first_formula.is_optimized)
        print(first_formula)

    def test_plus(self):
        m_of_n = self.knowledge[21]
        print(m_of_n)
        m_of_n.optimize()
        self.assertTrue(m_of_n.is_optimized)
        print(m_of_n)

    def test_mix(self):
        pyramidine_rich = self.knowledge[22]
        print(pyramidine_rich)
        pyramidine_rich.optimize()
        self.assertTrue(pyramidine_rich.is_optimized)
        print(pyramidine_rich)


if __name__ == '__main__':
    unittest.main()
