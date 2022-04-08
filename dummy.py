from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense
from logic.prolog import EnricherFuzzifier, PrologFormula
from ski.injectors import DataEnricher
from test.resources.rules.prolog import PATH, get_rules

mapping: dict[str, int] = {'setosa': 0, 'virginica': 1, 'versicolor': 2}
kb_path = str(PATH / 'iris-kb')
q_path = str(PATH / 'iris-q')

input_layer = Input(4)
x = Dense(16, activation='relu')(input_layer)
x = Dense(16, activation='relu')(x)
x = Dense(3, activation='softmax')(x)
predictor = Model(input_layer, x)

x, y = load_iris(return_X_y=True, as_frame=True)
encoder = OneHotEncoder(sparse=False)
encoder.fit_transform([y])
dataset = x.join(y)
train, test = train_test_split(dataset, test_size=0.5, random_state=0)
train_x, train_y = train.iloc[:, :-1], train.iloc[:, -1]
test_x, test_y = test.iloc[:, :-1], test.iloc[:, -1]

fuzzifier = EnricherFuzzifier(kb_path + '.txt')
injector = DataEnricher(predictor, train_x, fuzzifier, mapping)
queries = [PrologFormula(rule) for rule in get_rules(q_path)]
new_predictor = injector.inject(queries)

new_predictor.compile('adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
new_predictor.fit(train_x, train_y, batch_size=4, epochs=30)
accuracy = new_predictor.evaluate(test_x, test_y)[1]
print(accuracy)

