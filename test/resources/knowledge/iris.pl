iris(PetalLength, PetalWidth, SepalLength, SepalWidth, virginica) :- \+('>='(PetalWidth, 0.664341), '=<'(PetalWidth, 1.651423)).
iris(PetalLength, PetalWidth, SepalLength, SepalWidth, setosa) :- '=<'(PetalWidth, 1.651423).
iris(PetalLength, PetalWidth, SepalLength, SepalWidth, versicolor) :- true.