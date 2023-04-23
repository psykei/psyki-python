import unittest
from test.resources.data import SpliceJunction


class TestFormulaOptimization(unittest.TestCase):
    knowledge = SpliceJunction.get_knowledge()

    def test_conjunction(self):
        first_formula = self.knowledge[0]
        first_formula.optimize()
        self.assertTrue(first_formula.is_optimized)

    def test_plus(self):
        m_of_n = self.knowledge[18]
        m_of_n.optimize()
        self.assertTrue(m_of_n.is_optimized)

    def test_mix(self):
        pyramidine_rich = self.knowledge[22]
        pyramidine_rich.optimize()
        self.assertTrue(pyramidine_rich.is_optimized)


if __name__ == "__main__":
    unittest.main()
