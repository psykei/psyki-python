import unittest
import numpy as np
from psyki.fuzzifiers import Fuzzifier
from tensorflow.python.ops.numpy_ops import argmax
from tensorflow import constant, float32, reshape, cast, stack, assert_equal, tile
from tensorflow.python.ops.array_ops import gather_nd
from test.resources.knowledge import PATH as KNOWLEDGE_PATH
from psyki.logic.prolog import TuProlog
from test.resources.data import get_dataset, SpliceJunction, get_splice_junction_processed_dataset, Poker


class TestLukasiewiczSimple(unittest.TestCase):
    knowledge = TuProlog.from_file(KNOWLEDGE_PATH / 'simple.pl').formulae
    fuzzifier = Fuzzifier.get('lukasiewicz')([{'no': 0, 'yes': 1}, {'X': 0, 'Y': 1}])
    functions = fuzzifier.visit(knowledge)
    predicted_output_yes = constant([0, 1], dtype=float32)
    predicted_output_yes = reshape(predicted_output_yes, [1, 2])
    predicted_output_no = constant([1, 0], dtype=float32)
    predicted_output_no = reshape(predicted_output_no, [1, 2])
    true = tile(reshape(constant(0.), [1, 1]), [1, 1])
    false = tile(reshape(constant(1.), [1, 1]), [1, 1])

    def test_greater_yes(self):
        function_yes = self.functions['yes']
        function_no = self.functions['no']
        input_values = constant([3.4, 1.7], dtype=float32, shape=[1, 2])

        # Functions must output 0 (true) for both yes and no classes, because the prediction is correct.
        actual_output_yes = function_yes(input_values, self.predicted_output_yes)
        actual_output_no = function_no(input_values, self.predicted_output_yes)
        assert_equal(self.true, actual_output_no)
        assert_equal(self.true, actual_output_yes)

        # Functions must output 0 (true) for yes and 1 (false) for no, because the prediction is wrong.
        actual_output_yes = function_yes(input_values, self.predicted_output_no)
        actual_output_no = function_no(input_values, self.predicted_output_no)
        assert_equal(self.false, actual_output_no)
        assert_equal(self.true, actual_output_yes)

    def test_greater_no(self):
        function_yes = self.functions['yes']
        function_no = self.functions['no']
        input_values = constant([-2.2, 5.7], dtype=float32, shape=[1, 2])

        # Functions must output 0 (true) for both yes and no classes, because the prediction is correct.
        actual_output_yes = function_yes(input_values, self.predicted_output_no)
        actual_output_no = function_no(input_values, self.predicted_output_no)
        assert_equal(self.true, actual_output_no)
        assert_equal(self.true, actual_output_yes)

        # Functions must output 1 (false) for yes and 0 (true) for no, because the prediction is wrong.
        actual_output_yes = function_yes(input_values, self.predicted_output_yes)
        actual_output_no = function_no(input_values, self.predicted_output_yes)
        assert_equal(self.true, actual_output_no)
        assert_equal(self.false, actual_output_yes)


class TestLukasiewiczOnSpliceJunction(unittest.TestCase):
    knowledge = TuProlog.from_file(KNOWLEDGE_PATH / 'splice-junction.pl').formulae
    fuzzifier = Fuzzifier.get('lukasiewicz')([SpliceJunction.class_mapping, SpliceJunction.feature_mapping])
    functions = fuzzifier.visit(knowledge)

    def test_on_dataset(self):
        data = get_splice_junction_processed_dataset('splice-junction-data.csv')
        x, y = data.iloc[:, :-1], data.iloc[:, -1:]
        y = np.eye(3)[y.astype(int)].reshape([y.shape[0], 3])
        x, y = cast(x, dtype=float32), cast(y, dtype=float32)
        functions = [self.functions[name] for name, _ in sorted(SpliceJunction.class_mapping.items(), key=lambda i: i[1])]
        result = stack([reshape(function(x, y), [x.shape[0], 1]) for function in functions], axis=1)
        # Per class errors using the provided knowledge
        #         IE    EI     N
        #   IE   295     0   473    ->  errors = 473
        #   EI    25    31   711    ->  errors = 25 + 711 = 736
        #   N      3     0  1652    ->  errors = 3
        self.assertTrue(np.all(sum(result) == constant([736, 473, 3], dtype=float32, shape=[3, 1])))


class TestLukasiewiczOnPoker(unittest.TestCase):
    knowledge = TuProlog.from_file(KNOWLEDGE_PATH / 'poker.pl').formulae
    fuzzifier = Fuzzifier.get('lukasiewicz')([Poker.class_mapping, Poker.feature_mapping])
    functions = fuzzifier.visit(knowledge)
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
        hand1 = constant([4, 10, 4, 11, 4, 1, 4, 13, 4, 12], dtype=float32)
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
        poker_training = get_dataset('poker-train.csv')
        functions = [self.functions[name] for name, _ in sorted(Poker.class_mapping.items(), key=lambda i: i[1])]
        train_x = poker_training.iloc[:, :-1]
        train_y = poker_training.iloc[:, -1]
        train_y = np.eye(10)[train_y.astype(int)]
        x, y = cast(train_x, dtype=float32), cast(train_y, dtype=float32)
        result = stack([reshape(function(x, y), [x.shape[0], 1]) for function in functions], axis=1)
        indices = stack([range(0, x.shape[0]), argmax(train_y, axis=1)], axis=1)
        assert_equal(gather_nd(result, indices), 0.)


if __name__ == '__main__':
    unittest.main()
