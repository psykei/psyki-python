<h1 align="center"> PSyKI </h1>
<h2 align="center">Platform for Symbolic Knowledge Injection</h2>

[![PyPI version](https://badge.fury.io/py/psyki.svg)](https://badge.fury.io/py/psyki)
[![Test](https://github.com/psykei/psyki-python/actions/workflows/check.yml/badge.svg?event=push)]()
![Codecov](https://img.shields.io/codecov/c/github/psykei/psyki-python)
[![Release](https://github.com/psykei/psyki-python/actions/workflows/deploy.yml/badge.svg?event=push)]()
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Stand With Ukraine](https://raw.githubusercontent.com/vshymanskyy/StandWithUkraine/main/badges/StandWithUkraine.svg)](https://stand-with-ukraine.pp.ua)

Some quick links:
<!-- * [Home Page](https://apice.unibo.it/xwiki/bin/view/PSyKI/) -->
* [GitHub Repository](https://github.com/psykei/psyki-python)
* [PyPi Repository](https://pypi.org/project/psyki/)
* [Issues](https://github.com/psykei/psyki-python/issues)

### Reference paper

> Matteo Magnini, Giovanni Ciatto, Andrea Omicini. "[On the Design of PSyKI: A Platform for Symbolic Knowledge Injection into Sub-Symbolic Predictors]", in: Proceedings of the 4th International Workshop on EXplainable and TRAnsparent AI and Multi-Agent Systems, 2022.

Bibtex: 
```bibtex
@incollection{psyki-extraamas2022,
    author = {Magnini, Matteo and Ciatto, Giovanni and Omicini, Andrea},
    booktitle = {Explainable and Transparent AI and Multi-Agent Systems},
    chapter = 6,
    dblp = {conf/atal/MagniniCO22},
    doi = {10.1007/978-3-031-15565-9_6},
    editor = {Calvaresi, Davide and Najjar, Amro and Winikoff, Michael and Främling, Kary},
    eisbn = {978-3-031-15565-9},
    eissn = {1611-3349},
    iris = {11585/899511},
    isbn = {978-3-031-15564-2},
    issn = {0302-9743},
    keywords = {Symbolic Knowledge Injection, Explainable AI, XAI, Neural Networks, PSyKI},
    note = {4th International Workshop, EXTRAAMAS 2022, Virtual Event, May 9--10, 2022, Revised Selected Papers},
    pages = {90--108},
    publisher = {Springer},
    scholar = {7587528289517313138},
    scopus = {2-s2.0-85138317005},
    series = {Lecture Notes in Computer Science},
    title = {On the Design of {PSyKI}: a Platform for Symbolic Knowledge Injection into Sub-Symbolic Predictors},
    url = {https://link.springer.com/chapter/10.1007/978-3-031-15565-9_6},
    urlpdf = {https://link.springer.com/content/pdf/10.1007/978-3-031-15565-9_6.pdf},
    volume = 13283,
    wos = {000870042100006},
    year = 2022
}
```

## Intro

PSyKI (<u><b>P</b></u>latform for <u><b>Sy</b></u>mbolic <u><b>K</b></u>nowledge <u><b>I</b></u>njection) is a python library for symbolic knowledge injection (<b>SKI</b>).
SKI is a particular subclass of neuro-symbolic (<b>NeSy</b>) integration techniques.
PSyKI offers SKI algorithms (a.k.a. <b>injectors</b>) along with quality of service metrics (<b>QoS</b>).

### More in detail

An `Injector` is a SKI algorithm that may -- or may not -- take a sub-symbolic predictor in conjunction with prior symbolic knowledge to create a new predictor through the `inject` method.
We refer to the new predictor as **educated**, while predictors that are not affected by symbolic knowledge are called **uneducated**.

Knowledge can be represented in many ways.
The most common is the representation via logic formulae.
PSyKI integrates [`2ppy`](https://github.com/tuProlog/2ppy), a python porting of [`2p-kt`](https://github.com/tuProlog/2p-kt) (a multi-paradigm logic programming framework).
Thanks to this integration, PSyKI supports logic formulae written with the formalism of Prolog.
Therefore, all subsets of the Prolog language (including Prolog itself) are potentially supported (e.g., propositional logic, Datalog, etc.).
It is worth noting that each injector may have its own requirements on the knowledge representation.

List of available injectors:

 - [`KBANN`](https://www.sciencedirect.com/science/article/pii/0004370294901058), one of the first injector introduced in literature;
 - [`KILL`](http://ceur-ws.org/Vol-3261/paper5.pdf), constrains a NN by altering its predictions using the knowledge;
 - [`KINS`](http://ceur-ws.org/Vol-3204/paper_25.pdf), structure the knowledge adding ad-hoc layers into a NN.

### High level architecture

![PSyKI class diagram](https://www.plantuml.com/plantuml/svg/TLF1Rjim3BthAnvyweQvGA_1C7Gx32YMvTBUYc9WPDdH8ec1H6z8i_pxP3bnd64z9NmKtoCVwVia5ANtJgcqjM57aJoS3KRsEmEEic6bTgItr1auxgm-A0NGEaaaBVZAqVUE3XbJm541WSLWpIBimUtvWGA0XeIG2tijVJG5QZc2HcB4tWsW2KqXKOEGTfGIdZQ6u_vGAfnDydnYVS4sy6zdciwC0bRBSnRu01la1QsXGUY7fzt_qeNxb3mgXPCCghiAx-iQLQYczjNnOaCswjg4X_3JQE5O6lpEZN7OHJEeSHoWHube-zTNsrfJ05iARavwKdxUBSRIkOtHTXi1jvF2Od55Z3wOfjSaffaRD_dsxM7rEBfcWy3HliWVvm-MoyCy_l9vjHehMiSaO6ywciKTUK_p5gjDFfHObyCnOc82jyD48DTnjBBngG8bhEuKHdhStfQeT3S6fG4RjSjyAC-rmZGqFlwfwu9erALg_3lIJV1oURMboV3qpyMUyN5C_BB9oiqRLvMNGc7_ncNFDugdI26rcI0XxVsQtUcWqzb-1Y7rwthANdyDc2smp74vnkpHfyaCTN4bMvUpipwKkiyKlNT_0G00)
*Class diagram representing the relations between `Injector`, `Theory` and `Fuzzifier` classes*

<!--
To generate/edit the class diagram browse the URL above, after replacing `svg` with `uml`
-->

The core abstractions of PSyKI are the following:

 - `Injector`: a SKI algorithm;
 - `Theory`: symbolic knowledge plus additional information about the domain;
 - `Fuzzifier`: entity that transforms (fuzzify) symbolic knowledge into a sub-symbolic data structure.

The class `Theory` is built upon the symbolic knowledge and the metadata of the dataset (extracted by a Pandas DataFrame).
The knowledge can be generated by an adapter that parses the Prolog theory (e.g., a `.pl` file, a string) and generates a list of `Formula` objects.
Each `Injector` has one `Fuzzifier`.
The `Fuzzifier` is used to transform the `Theory` into a sub-symbolic data structure (e.g., ad-hoc layers of a NN).
Different fuzzifiers encode the knowledge in different ways.

To avoid confusion, we use the following terminology:

 - **rule** is a single logic clause;
 - **knowledge** is the set of rules;
 - **theory** is the knowledge plus metadata.

### Hello world

The following example shows how to use PSyKI to inject knowledge into a NN.

```python
import pandas as pd
from tensorflow.keras.models import Model
from psyki.logic import Theory
from psyki.ski import Injector


def create_uneducated_predictor() -> Model:
...

dataset = pd.read_csv("path_to_dataset.csv")  # load dataset
knowledge_file = "path_to_knowledge_file.pl"  # load knowledge
theory = Theory(knowledge_file, dataset)      # create a theory
uneducated = create_uneducated_predictor()    # create a NN
injector = Injector.kins(uneducated)          # create an injector
educated = injector.inject(theory)            # inject knowledge into the NN

# From now on you can use the educated predictor as you would use the uneducated one
```

For more detailed examples, please refer to the demos in the [demo-psyki-python](https://github.com/psykei/demo-psyki-python) repository.

---------------------------------------------------------------------------------

##  Injection Assessment 🎯

Knowledge injection methods aim to enhance the sustainability of machine learning by minimizing the learning time required. This accelerated learning can possibly lead to improvements in model accuracy as well. Additionally, by incorporating prior knowledge, the data requirements for effective training may be loosened, allowing for smaller datasets to be utilized. Injecting knowledge has also the potential to increase model interpretability by preventing the predictor from becoming a black-box.

Most existing works on knowledge injection techniques showcase standard evaluation metrics that aim to quantify these potential benefits. However, accurately quantifying sustainability, accuracy improvements, dataset needs, and interpretability in a consistent manner remains an open challenge.

## Efficiency metrics: 

- **Memory footprint**, i.e., the size of the predictor under examination;
- **Data efficiency**, i.e., the amount of data required to train the predictor;
- **Latency**, i.e., the time required to run a predictor for inference;
- **Energy consumption**, i.e., the amount of energy required to train/run the predictor;

### <b>💾 Memory Footprint </b>
<u><i>Intuition:</i></u> injected knowledge can reduce the amount of notions to be learnt data-drivenly.

<u><i>Idea:</i></u> measure the amount of total operations (FLOPs or MACs) required by the model.

### <b>🗂️ Data Efficiency </b>
<u><i>Intuition:</i></u> several concepts are injected → some portions of training data are not required.

<u><i>Idea:</i></u> reducing the size of the training set $D$ for the educated predictor by letting the input knowledge $K$ compensate for such lack of data.

### <b>⏳ Latency </b>
<u><i>Intuition:</i></u> knowledge injection remove unnecessary computations required to draw a prediction.

<u><i>Idea:</i></u> measures the average time required to draw a single prediction from a dataset $T$.

### <b>⚡ Energy Consumption </b> 
<u><i>Intuition:</i></u> knowledge injection reduces learning complexity → reduces amount of computations for training and running a model.

<u><i>Idea:</i></u> i) measures the average energy consumption (mW) for a single update, on a training dataset $T$; ii) measures the average energy consumption (mW) of a single forward on a test dataset $T$ composed by several samples.

## Example
The following example shows how to use QoS metrics.

```python

from psyki.qos.energy import Energy

# Model training
model1 = ... # create model 1
model2 = ... # create model 2

X_train, y_train = ... # get training data

model1.fit(X_train, y_train) 
model2.fit(X_train, y_train)

# Compute energy
params = {'x': X_train, 'y': y_train, 'epochs': 100}
training_energy = Energy.compute_during_training(model1, model2, params)


# Model inference
X_test, y_test = ... # get test data 

model1_preds = model1.predict(X_test)
model2_preds = model2.predict(X_test)

# Compute energy
params = {'x': X_test}  
inference_energy = Energy.compute_during_inference(model1, model2, params)

# The process for the other metrics (data efficiency, latency, and memory) is the same.
```

### Reference paper

> Andrea Agiollo, Andrea Rafanelli, Matteo Magnini, Giovanni Ciatto, Andrea Omicini. "[Symbolic knowledge injection meets intelligent agents: QoS metrics and experiments]", in:  Autonomous Agents and Multi-Agent Systems, 2023.

Bibtex: 
```bibtex
@article{ski-qos23,
  author       = {Andrea Agiollo and
                  Andrea Rafanelli and
                  Matteo Magnini and
                  Giovanni Ciatto and
                  Andrea Omicini},
  title        = {Symbolic knowledge injection meets intelligent agents: QoS metrics
                  and experiments},
  journal      = {Auton. Agents Multi Agent Syst.},
  volume       = {37},
  number       = {2},
  pages        = {27},
  year         = {2023},
  url          = {https://doi.org/10.1007/s10458-023-09609-6},
  doi          = {10.1007/S10458-023-09609-6},
  timestamp    = {Tue, 12 Sep 2023 07:57:44 +0200},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
------------------------------------------------------------------
## Users

PSyKI is deployed as a library on Pypi, and it can therefore be installed as Python package by running:
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


## Developers

Working with PSyKI codebase requires a number of tools to be installed:
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
If you meet some problem in using or developing PSyKI, you are encouraged to signal it through the project
["Issues" section](https://github.com/psykei/psyki-python/issues) on GitHub.

