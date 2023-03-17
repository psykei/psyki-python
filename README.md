# PSyKI

Some quick links:
<!-- * [Home Page](https://apice.unibo.it/xwiki/bin/view/PSyKI/) -->
* [GitHub Repository](https://github.com/psykei/psyki-python)
* [PyPi Repository](https://pypi.org/project/psyki/)
* [Issues](https://github.com/psykei/psyki-python/issues)

### Reference paper

> Matteo Magnini, Giovanni Ciatto, Andrea Omicini. "[On the Design of PSyKI: A Platform for Symbolic Knowledge Injection into Sub-Symbolic Predictors]", in: Proceedings of the 4th International Workshop on EXplainable and TRAnsparent AI and Multi-Agent Systems, 2022.

Bibtex: 
```bibtex
@inproceedings{PsykiExtraamas2022,
	keywords = {Symbolic Knowledge Injection,  Explainable AI, XAI, Neural Networks, PSyKI},
	year = 2022,
	talk = {Talks.PsykiExtraamas2022},
	author = {Magnini, Matteo and Ciatto, Giovanni and Omicini, Andrea},
	venue_e = {Events.Extraamas2022},
	sort = {inproceedings},
	publisher = {Springer},
	status = {In press},
	title = {On the Design of PSyKI: a Platform for Symbolic Knowledge Injection into Sub-Symbolic Predictors},
	booktitle = {Proceedings of the 4th International Workshop on EXplainable and TRAnsparent AI and Multi-Agent Systems}
}
```

## Intro

PSyKI (Platform for Symbolic Knowledge Injection) is a library for **symbolic knowledge injection** (SKI).
In the literature, SKI may also be referred as **neuro-symbolic integration**.
PSyKI offers SKI algorithms (**injectors**) along with quality of service metrics (QoS) and other utility functionalities.
Finally, the library is open to extendability.

### More in detail

An `Injector` is a SKI algorithm that may -- or may not -- take a sub-symbolic predictor in conjunction with prior symbolic knowledge to create a new predictor through the `inject` method.
We refer to the new predictor as **educated**, while predictors that do not include symbolic knowledge are called **uneducated**.

Knowledge can be represented in many ways.
The most common is the representation via logic formulae.
PSyKI integrates [`2ppy`](https://github.com/tuProlog/2ppy), a python porting of [`2p-kt`](https://github.com/tuProlog/2p-kt) (a multi-paradigm logic programming framework).
Thanks to this integration, PSyKI supports logic formulae written with the formalism of Prolog.
Therefore, all subsets of the Prolog language (including Prolog itself) are potentially supported (e.g., propositional logic, Datalog, etc.).

<!---
![PSyKE class diagram](https://www.plantuml.com/plantuml/svg/XP9FJy904CNl_HIJdXJ3mIlnW68q9iIOS3GUXktIZDq_PFz80Fdkrcu7Q6tmqdJVlDtNz-jEVK0NebRP6aM5fGHV4Uop381Ca6w5GiAB-PGYMBUlLO0RM3jPqAymWJT-RKVKMAzMrkceK4vWJZwyFwNbntLtmw4RqxhsVJdkThGYUOp_8a-N8kxDzk_XCsjyS4Y6J7awyT3nB8AB8aJNizGUeT1xcADU5ZZ7RLT-bM5ZNJMpUcqzZvWPZ24VvTmjo-eXarQs995OirHWm5fEuaak7MgDBG2EVKpU4pM0jlNnUsE5dkI6nAuPi_vi2wawoY8ksnOFaLJ19G2Qnx7BtVTutA_RE1XNiWp37Z8oyRvlw0JUSzRg7sigsfY6WfAdNptbjrIJfKjw0CdPnMYcPsiOrVFyDh_0f0UKTmp3uPlDLsC4zQywP5DfYgZL3m00)
-->
<!--
To generate/edit the class diagram browse the URL above, after replacing `svg` with `uml`
-->

List of available injectors:

 - [`KBANN`](http://www.aaai.org/Library/AAAI/1990/aaai90-129.php), one of the first injector introduced in literature;
 - [`KILL`](http://ceur-ws.org/Vol-3261/paper5.pdf), constrains a NN by altering its predictions using the knowledge;
 - [`KINS`](http://ceur-ws.org/Vol-3204/paper_25.pdf), structure the knowledge adding ad-hoc layers into a NN.

## Users

PSyKI is deployed as a library on Pypi, and it can therefore be installed as Python pachage by running:
```text
pip install psyki
```

### Requirements

- python 3.9+
- java 11
- 2ppy 0.4.0
- tensorflow 2.7.0
- numpy 1.22.3
- scikit-learn 1.0.2
- pandas 1.4.2
- codecarbon 2.1.4

<!---
### Examples

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
-->

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

