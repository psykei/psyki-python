import unittest
from psyki.logic.datalog.grammar.adapters.antlr4 import get_formula_from_string


class TestFormula(unittest.TestCase):
    virginica_rule = "class(PL,PW,SL,SW,virginica) ← PL > 2.28 ∧ PW > 1.64"
    expected_formula_structure = "class(PL,PW,SL,SW,virginica)←((((PL)>(2.28)))∧(((PW)>(1.64))))"

    def test_parsing_with_antlr4(self):
        formula = get_formula_from_string(self.virginica_rule)
        self.assertEqual(self.expected_formula_structure, str(formula))
