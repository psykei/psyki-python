import unittest
import numpy as np
from psyki.logic import Fuzzifier
from tensorflow.python.ops.numpy_ops import argmax
from tensorflow import constant, float32, reshape, cast, stack, assert_equal, tile
from tensorflow.python.ops.array_ops import gather_nd
from psyki.logic.datalog.grammar.adapters.antlr4 import get_formula_from_string
from test.resources.data import get_dataset
from test.resources.rules import get_rules
from test.resources.rules.poker import FEATURE_MAPPING as POKER_FEATURE_MAPPING, CLASS_MAPPING as POKER_CLASS_MAPPING


class TestLukasiewicz(unittest.TestCase):

    rules = list(get_rules('poker'))
    formulae = [get_formula_from_string(rule) for rule in rules]
    fuzzifier = Fuzzifier.get('lukasiewicz')([POKER_CLASS_MAPPING, POKER_FEATURE_MAPPING])
    functions = fuzzifier.visit(formulae)
    true = tile(reshape(constant(0.), [1, 1]), [1, 1])
    false = tile(reshape(constant(1.), [1, 1]), [1, 1])
    
    def test_nothing(self):
        hand1 = constant([2, 6, 2, 1, 4, 13, 2, 4, 4, 9], dtype=float32)
        hand2 = constant([4, 9, 3, 10, 4, 7, 4, 9, 3, 8], dtype=float32)
        output1 = constant([1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float32)
        output2 = constant([0, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float32)
        function = self.functions['nothing']

        self._test_implication_hand_output_combinations(function, hand1, hand2, output1, output2)

    def test_two(self):
        hand1 = constant([4, 9, 2, 2, 4, 2, 4, 6, 3, 9], dtype=float32)
        hand2 = constant([4, 1, 2, 2, 4, 7, 4, 10, 3, 9], dtype=float32)
        output1 = constant([0, 0, 1, 0, 0, 0, 0, 0, 0, 0], dtype=float32)
        output2 = constant([0, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=float32)
        function = self.functions['two']

        self._test_implication_hand_output_combinations(function, hand1, hand2, output1, output2)

    def test_flush(self):
        hand1 = constant([4, 4, 4, 13, 4, 7, 4, 11, 4, 1], dtype=float32)
        hand2 = constant([4, 4, 1, 13, 4, 7, 4, 11, 4, 1], dtype=float32)
        output1 = constant([0, 0, 0, 0, 0, 1, 0, 0, 0, 0], dtype=float32)
        output2 = constant([0, 0, 0, 0, 0, 0, 0, 1, 0, 0], dtype=float32)
        function = self.functions['flush']

        self._test_implication_hand_output_combinations(function, hand1, hand2, output1, output2)

    def test_full(self):
        hand1 = constant([3, 2, 1, 2, 3, 11, 1, 11, 4, 11], dtype=float32)
        hand2 = constant([4, 1, 4, 2, 4, 7, 4, 10, 4, 9], dtype=float32)
        output1 = constant([0, 0, 0, 0, 0, 0, 1, 0, 0, 0], dtype=float32)
        output2 = constant([0, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=float32)
        function = self.functions['full']

        self._test_implication_hand_output_combinations(function, hand1, hand2, output1, output2)

    def test_four(self):
        hand1 = constant([4, 9, 1, 9, 4, 7, 2, 9, 3, 9], dtype=float32)
        hand2 = constant([4, 9, 4, 5, 4, 7, 2, 9, 3, 9], dtype=float32)
        output1 = constant([0, 0, 0, 0, 0, 0, 0, 1, 0, 0], dtype=float32)
        output2 = constant([0, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=float32)
        function = self.functions['four']

        self._test_implication_hand_output_combinations(function, hand1, hand2, output1, output2)

    def test_three(self):
        hand1 = constant([4, 9, 4, 2, 4, 7, 3, 9, 1, 9], dtype=float32)
        hand2 = constant([4, 1, 4, 2, 4, 7, 4, 10, 1, 9], dtype=float32)
        output1 = constant([0, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=float32)
        output2 = constant([0, 0, 0, 1, 0, 0, 0, 1, 0, 0], dtype=float32)
        function = self.functions['three']

        self._test_implication_hand_output_combinations(function, hand1, hand2, output1, output2)

    def test_pair(self):
        hand1 = constant([4, 9, 4, 2, 4, 7, 4, 6, 2, 9], dtype=float32)
        hand2 = constant([4, 1, 4, 2, 4, 7, 4, 10, 2, 9], dtype=float32)
        output1 = constant([0, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float32)
        output2 = constant([0, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=float32)
        function = self.functions['pair']

        self._test_implication_hand_output_combinations(function, hand1, hand2, output1, output2)

    def test_straight(self):
        hand1 = constant([1, 9, 4, 10, 2, 7, 4, 6, 3, 8], dtype=float32)
        hand2 = constant([1, 1, 4, 2, 2, 7, 4, 10, 3, 9], dtype=float32)
        output1 = constant([0, 0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=float32)
        output2 = constant([0, 0, 0, 0, 0, 0, 1, 0, 0, 0], dtype=float32)
        function = self.functions['straight']

        self._test_implication_hand_output_combinations(function, hand1, hand2, output1, output2)

        # Straight is also 10, 11, 12, 13, 1!
        hand3 = constant([1, 1, 4, 11, 2, 13, 4, 10, 3, 12], dtype=float32)
        hand3 = tile(reshape(hand3, [1, 10]), [1, 1])
        output1 = tile(reshape(output1, [1, 10]), [1, 1])
        result = reshape(function(hand3, output1), [1, 1])
        assert_equal(result, self.true)

    def test_straight_flush(self):
        hand1 = constant([4, 9, 4, 10, 4, 7, 4, 6, 4, 8], dtype=float32)
        hand2 = constant([4, 9, 3, 10, 4, 7, 4, 6, 3, 8], dtype=float32)
        output1 = constant([0, 0, 0, 0, 0, 0, 0, 0, 1, 0], dtype=float32)
        output2 = constant([0, 0, 0, 0, 0, 0, 1, 0, 0, 0], dtype=float32)
        function = self.functions['straight_flush']

        self._test_implication_hand_output_combinations(function, hand1, hand2, output1, output2)

    def test_royal_flush(self):
        hand1 = constant([1, 10, 1, 11, 1, 13, 1, 12, 1, 1], dtype=float32)
        hand2 = constant([1, 9, 1, 11, 1, 13, 1, 10, 1, 12], dtype=float32)
        output1 = constant([0, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=float32)
        output2 = constant([0, 0, 0, 0, 0, 0, 0, 0, 1, 0], dtype=float32)
        function = self.functions['royal_flush']

        self._test_implication_hand_output_combinations(function, hand1, hand2, output1, output2)

    def _test_implication_hand_output_combinations(self, function, hand1, hand2, output1, output2) -> None:
        result1, result2, result3, result4 = self._get_combination_values(function, hand1, hand2, output1, output2)
        assert_equal(result1, self.true)
        assert_equal(result2, self.false)
        assert_equal(result3, self.true)
        assert_equal(result4, self.true)

    @staticmethod
    def _get_combination_values(function, hand1, hand2, output1, output2):
        hand1 = tile(reshape(hand1, [1, 10]), [1, 1])
        hand2 = tile(reshape(hand2, [1, 10]), [1, 1])
        output1 = tile(reshape(output1, [1, 10]), [1, 1])
        output2 = tile(reshape(output2, [1, 10]), [1, 1])
        result1 = reshape(function(hand1, output1), [1, 1])
        result2 = reshape(function(hand2, output1), [1, 1])
        result3 = reshape(function(hand1, output2), [1, 1])
        result4 = reshape(function(hand2, output2), [1, 1])
        return result1, result2, result3, result4

    def test_on_dataset(self):
        poker_training = get_dataset('poker', 'train')
        functions = [self.functions[name] for name, _ in sorted(POKER_CLASS_MAPPING.items(), key=lambda i: i[1])]
        train_x = poker_training[:, :-1]
        train_y = poker_training[:, -1]
        train_y = np.eye(10)[train_y.astype(int)]
        x, y = cast(train_x, dtype=float32), cast(train_y, dtype=float32)
        result = stack([reshape(function(x, y), [x.shape[0], 1]) for function in functions], axis=1)
        indices = stack([range(0, len(poker_training)), argmax(train_y, axis=1)], axis=1)
        assert_equal(gather_nd(result, indices), 0.)


if __name__ == '__main__':
    unittest.main()
