import unittest
from tensorflow.python.framework.random_seed import set_seed
from psyki.logic.prolog import TuProlog
from psyki.qos.data import DataEfficiency
from psyki.ski import Injector
from test.psyki.qos import split_dataset, evaluate_metric
from test.resources.data import get_splice_junction_processed_dataset, SpliceJunction
from test.resources.knowledge import PATH as KNOWLEDGE_PATH
from test.utils import create_standard_fully_connected_nn


class TestDataOnSplice(unittest.TestCase):
    seed = 0
    formulae = TuProlog.from_file(str(KNOWLEDGE_PATH / 'splice-junction.pl')).formulae
    dataset = get_splice_junction_processed_dataset('splice-junction-data.csv')
    dataset1 = split_dataset(dataset)
    dataset2 = split_dataset(dataset, test_size=0.5)
    model = create_standard_fully_connected_nn(input_size=240, output_size=3, layers=3, neurons=128)

    def test_data_fit_with_kins(self):
        print('TEST ENERGY FIT WITH KINS ON SPLICE JUNCTION')
        set_seed(self.seed)
        additional_params = {
            'seed': self.seed,
            'epochs1': 100,
            'epochs2': 100,
            'train_x1': self.dataset1['train_x'],
            'train_y1': self.dataset1['train_y'],
            'test_x1': self.dataset1['test_x'],
            'test_y1': self.dataset1['test_y'],
            'train_x2': self.dataset2['train_x'],
            'train_y2': self.dataset2['train_y'],
            'test_x2': self.dataset2['test_x'],
            'test_y2': self.dataset2['test_y'],
        }
        injector = Injector.kins(self.model, feature_mapping=SpliceJunction.feature_mapping)
        educated_predictor = injector.inject(self.formulae)
        data_efficiency = evaluate_metric(self.model, educated_predictor, self.dataset1, DataEfficiency.compute_during_training, additional_params)
        print(data_efficiency)
        self.assertTrue(isinstance(data_efficiency, float))

    def test_data_inf_with_kins(self):
        print('TEST ENERGY FIT WITH KBANN ON SPLICE JUNCTION')
        set_seed(self.seed)
        additional_params = {
            'seed': self.seed,
            'metric1': 0.93,
            'metric2': 0.95,
            'epochs1': 100,
            'epochs2': 80,
            'train_x1': self.dataset1['train_x'],
            'train_y1': self.dataset1['train_y'],
            'train_x2': self.dataset2['train_x'],
            'train_y2': self.dataset2['train_y'],
        }
        injector = Injector.kins(self.model, feature_mapping=SpliceJunction.feature_mapping)
        educated_predictor = injector.inject(self.formulae)
        data_efficiency = evaluate_metric(self.model, educated_predictor, self.dataset1, DataEfficiency.compute_during_inference, additional_params)
        print(data_efficiency)
        self.assertTrue(isinstance(data_efficiency, float))


if __name__ == '__main__':
    unittest.main()
