# PSyKI

Some quick links:
<!-- * [Home Page](https://apice.unibo.it/xwiki/bin/view/PSyKI/) -->
* [GitHub Repository](https://github.com/psykei/psyki-python)
* [PyPi Repository](https://pypi.org/project/psyki/)
* [Issues](https://github.com/psykei/psyki-python/issues)


## Intro

PSyKI (Platform for Symbolic Knowledge Injection) is a library for Symbolic Knowledge Injection (SKI) into sub-symbolic predictors.
PSyKI offers SKI algorithms (injectors), and it is open to extendability.

An `Injector` is a SKI algorithm that takes a sub-symbolic predictor and prior symbolic knowledge, and it creates a new predictor through method  `inject`.
Knowledge can be represented in many ways, the most common is the representation via textual logic formulae.
Currently, (stratified) Datalog formulae (allowing negation) are supported.
Knowledge in this form should be processed into a visitable data structure `Formula` that is specific w.r.t. the representation.
User can use the `Antlr` adapter to get proper `Formula` from the AST generated by antlr4.
Knowledge represented via `Formula` object can be embedded in a sub-symbolic form through a `Fuzzifier`.
A `Fuzzifier` is a visitor for `Formula` objects that outputs a sub-symbolic object that can be injected into a sub-symbolic predictor. 

![PSyKE class diagram](https://www.plantuml.com/plantuml/svg/XP9FJy904CNl_HIJdXJ3mIlnW68q9iIOS3GUXktIZDq_PFz80Fdkrcu7Q6tmqdJVlDtNz-jEVK0NebRP6aM5fGHV4Uop381Ca6w5GiAB-PGYMBUlLO0RM3jPqAymWJT-RKVKMAzMrkceK4vWJZwyFwNbntLtmw4RqxhsVJdkThGYUOp_8a-N8kxDzk_XCsjyS4Y6J7awyT3nB8AB8aJNizGUeT1xcADU5ZZ7RLT-bM5ZNJMpUcqzZvWPZ24VvTmjo-eXarQs995OirHWm5fEuaak7MgDBG2EVKpU4pM0jlNnUsE5dkI6nAuPi_vi2wawoY8ksnOFaLJ19G2Qnx7BtVTutA_RE1XNiWp37Z8oyRvlw0JUSzRg7sigsfY6WfAdNptbjrIJfKjw0CdPnMYcPsiOrVFyDh_0f0UKTmp3uPlDLsC4zQywP5DfYgZL3m00)

<!--
To generate/edit the class diagram browse the URL above, after replacing `svg` with `uml`
-->

Currently, implemented injectors are:

 - `LambdaLayer`, performs injection into NN of any shape via constraining;
 - `NetworkComposer`, performs injection into NN of any shape via structuring.


## Users

PSyKI is deployed as a library on Pypi, and it can therefore be installed as Python pachage by running:
```text
pip install psyki
```

### Requirements

- python 3.9+
- java 11 (for test and demo)
- antlr4-python3-runtime 4.9.3 (for test and demo)
- tensorflow 2.6.2
- numpy 1.19.2
- scikit-learn 1.0.1
- pandas 1.3.4

### Demo

`demo/demo.ipynb` is a notebook that shows how injection is applied to a network for poker hand classification task.
Rules are defined in `resources/rules/poker.csv`.


Example of injection:
```python
injector = NetworkComposer(model, feature_mapping)
predictor = injector.inject(formulae)
predictor.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
predictor.fit(train_x, train_y, verbose=1, batch_size=32, epochs=100)
```

Output:
```text
Epoch 1/100
782/782 [==============================] - 3s 906us/step - loss: 1.0029 - accuracy: 0.5090
Epoch 2/100
782/782 [==============================] - 1s 902us/step - loss: 0.9579 - accuracy: 0.5381
Epoch 3/100
782/782 [==============================] - 1s 899us/step - loss: 0.9447 - accuracy: 0.5451
Epoch 4/100
782/782 [==============================] - 1s 903us/step - loss: 0.9347 - accuracy: 0.5534
Epoch 5/100
782/782 [==============================] - 1s 896us/step - loss: 0.9249 - accuracy: 0.5547
Epoch 6/100
782/782 [==============================] - 1s 897us/step - loss: 0.9153 - accuracy: 0.5625
```

```python
loss, accuracy = predictor.evaluate(test_x, test_y)
print('Loss: ' + str(loss))
print('Accuracy: ' + str(accuracy))
```
Output:
```text
31250/31250 [==============================] - 26s 822us/step - loss: 0.0660 - accuracy: 0.9862
Loss: 0.06597686558961868
Accuracy: 0.9862030148506165
```

## Developers

Working with PSyKE codebase requires a number of tools to be installed:
* Python 3.9+
* JDK 11+ (please ensure the `JAVA_HOME` environment variable is properly configured)
* Git 2.20+

### Develop PSyKI with PyCharm

To participate in the development of PSyKI, we suggest the [PyCharm](https://www.jetbrains.com/pycharm/) IDE.

#### Importing the project

1. Clone this repository in a folder of your preference using `git_clone` appropriately
2. Open PyCharm
3. Select `Open`
4. Navigate your file system and find the folder where you cloned the repository
5. Click `Open`

### Developing the project

Contributions to this project are welcome. Just some rules:
* We use [git flow](https://github.com/nvie/gitflow), so if you write new features, please do so in a separate `feature/` branch
* We recommend forking the project, developing your stuff, then contributing back vie pull request
* Commit often
* Stay in sync with the `develop` (or `main | master`) branch (pull frequently if the build passes)
* Do not introduce low quality or untested code

#### Issue tracking
If you meet some problem in using or developing PSyKE, you are encouraged to signal it through the project
["Issues" section](https://github.com/psykei/psyki-python/issues) on GitHub.