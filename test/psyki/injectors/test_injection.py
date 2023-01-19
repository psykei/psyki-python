import unittest
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from psyki.logic import Formula, DefinitionFormula
from psyki.logic.prolog import TuProlog
from psyki.ski import Injector
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras import Input, Model
from tensorflow.python.framework.random_seed import set_seed
from psyki.ski.kill import LambdaLayer
from psyki.ski.kins import NetworkStructurer
from test.resources.data import get_dataset_dataframe, data_to_int, CLASS_MAPPING, AGGREGATE_FEATURE_MAPPING, \
    get_binary_data, get_splice_junction_extended_feature_mapping
from test.resources.knowledge import PATH
from test.utils import get_mlp, Conditions


class TestInjectionOnIris(unittest.TestCase):
    EPOCHS = 50
    BATCH_SIZE = 8
    VERBOSE = 0
    ACCEPTABLE_ACCURACY = 0.97

    set_seed(0)
    formulae = TuProlog.from_file(PATH / 'iris.pl').formulae
    input_layer = Input((4,))
    predictor = get_mlp(input_layer, 3, 3, 32, 'relu', 'softmax')
    predictor = Model(input_layer, predictor)
    x, y = load_iris(return_X_y=True, as_frame=True)
    encoder = OneHotEncoder(sparse=False)
    encoder.fit_transform([y])
    dataset = x.join(y)
    train, test = train_test_split(dataset, test_size=0.5, random_state=0)
    train_x, train_y = train.iloc[:, :-1], train.iloc[:, -1]
    test_x, test_y = test.iloc[:, :-1], test.iloc[:, -1]
    class_mapping = {'setosa': 0, 'virginica': 1, 'versicolor': 2}
    variable_mapping = {'SepalLength': 0, 'SepalWidth': 1, 'PetalLength': 2, 'PetalWidth': 3}

    def compile_and_train(self, model):
        model.compile('adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(self.train_x, self.train_y, batch_size=self.BATCH_SIZE, epochs=self.EPOCHS, verbose=self.VERBOSE)

    def test_lambda_layer(self):
        injector = LambdaLayer(self.predictor, self.class_mapping, self.variable_mapping, 'lukasiewicz')
        model = injector.inject(self.formulae)
        del injector
        # Test if clone is successful
        cloned_model = model.copy()

        self.compile_and_train(model)
        model = model.remove_constraints()
        model.compile('adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        accuracy = model.evaluate(self.test_x, self.test_y)[1]
        del model

        self.compile_and_train(cloned_model)
        cloned_model = cloned_model.remove_constraints()
        cloned_model.compile('adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        accuracy_cm = cloned_model.evaluate(self.test_x, self.test_y)[1]
        del cloned_model

        self.assertTrue(accuracy > self.ACCEPTABLE_ACCURACY)
        self.assertTrue(accuracy == accuracy_cm)

    def test_kins(self):
        injector = NetworkStructurer(self.predictor, self.variable_mapping, 'netbuilder', 2)
        model = injector.inject(self.formulae)
        del injector
        # Test if clone is successful
        cloned_model = model.copy()

        self.compile_and_train(model)
        accuracy = model.evaluate(self.test_x, self.test_y)[1]
        del model

        self.compile_and_train(cloned_model)
        accuracy_cm = cloned_model.evaluate(self.test_x, self.test_y)[1]
        del cloned_model

        self.assertTrue(accuracy > self.ACCEPTABLE_ACCURACY)
        self.assertTrue(accuracy == accuracy_cm)


class TestInjectionOnSpliceJunction(unittest.TestCase):
    EPOCHS = 100
    VERBOSE = 0

    set_seed(0)
    rules: list[Formula] = TuProlog.from_file(PATH / 'splice-junction.pl').formulae
    trainable = ['intron_exon', 'exon_intron', 'pyramidine_rich', 'class']
    for rule in rules:
        assert(isinstance(rule, DefinitionFormula))
        if rule.lhs.predication in trainable:
            rule.trainable = True

    data = get_dataset_dataframe('splice_junction')
    y = data_to_int(data.iloc[:, -1:], CLASS_MAPPING)
    x = get_binary_data(data.iloc[:, :-1], AGGREGATE_FEATURE_MAPPING)
    y.columns = [x.shape[1]]
    data = x.join(y)
    data, test = train_test_split(data, train_size=1000, random_state=0, stratify=data.iloc[:, -1])
    k_fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    train_indices, _ = list(k_fold.split(data.iloc[:, :-1], data.iloc[:, -1:]))[0]
    train = data.iloc[train_indices, :]
    train_x, train_y = train.iloc[:, :-1], train.iloc[:, -1:]
    early_stop = Conditions(train_x, train_y)
    test_x, test_y = test.iloc[:, :-1], test.iloc[:, -1:]

    input_layer = Input((4*60,))
    predictor = get_mlp(input_layer, 3, 3, [64, 32], 'relu', 'softmax', dropout=True)
    predictor = Model(input_layer, predictor)

    def common_test_function(self, injector: Injector, batch_size: int, acceptable_accuracy: float, constrain=False):
        model = injector.inject(self.rules)
        # Test if clone is successful
        # cloned_model = model.copy()
        del injector

        if constrain:
            loss = model.loss_function(SparseCategoricalCrossentropy())
        else:
            loss = 'sparse_categorical_crossentropy'

        model.compile('adam', loss=loss, metrics=['accuracy'])
        model.fit(self.train_x, self.train_y, batch_size=batch_size, epochs=self.EPOCHS, verbose=self.VERBOSE, callbacks=self.early_stop)
        accuracy = model.evaluate(self.test_x, self.test_y)[1]
        del model

        # if constrain:
        #     loss = cloned_model.loss_function(SparseCategoricalCrossentropy())
        # else:
        #     loss = 'sparse_categorical_crossentropy'
        #
        # cloned_model.compile('adam', loss=loss, metrics=['accuracy'])
        # cloned_model.fit(self.train_x, self.train_y, batch_size=batch_size, epochs=self.EPOCHS, verbose=self.VERBOSE, callbacks=self.early_stop)
        # accuracy_cm = cloned_model.evaluate(self.test_x, self.test_y)[1]
        # del cloned_model

        self.assertTrue(accuracy > acceptable_accuracy)
        # self.assertTrue(accuracy == accuracy_cm)

    def test_kbann(self):
        injector = Injector.kbann(self.predictor, get_splice_junction_extended_feature_mapping(), 'towell', 4, gamma=0)
        self.common_test_function(injector, batch_size=16, acceptable_accuracy=0.9)

    def test_kbann_with_constraining(self):
        injector = Injector.kbann(self.predictor, get_splice_junction_extended_feature_mapping(), 'towell', 4, gamma=10E-5)
        self.common_test_function(injector, batch_size=16, acceptable_accuracy=0.9, constrain=True)

    def test_kins(self):
        injector = Injector.kins(self.predictor, get_splice_junction_extended_feature_mapping())
        self.common_test_function(injector, batch_size=32, acceptable_accuracy=0.9)


if __name__ == '__main__':
    unittest.main()
