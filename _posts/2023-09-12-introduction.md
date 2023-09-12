---
title: Introduction
author: mmagnini
date: 2023-09-12 12:00:00 +0100
categories: [Technology, Documentation]
tags: [PSyKI]
img_path: /assets/img/
pin: true
math: true
mermaid: true
---

### What is PSyKI?

PSyKI (<u><b>P</b></u>latform for <u><b>Sy</b></u>mbolic <u><b>K</b></u>nowledge <u><b>I</b></u>njection) is a python library for symbolic knowledge injection (<b>SKI</b>).
SKI is a particular subclass of neuro-symbolic (<b>NeSy</b>) integration techniques.
PSyKI offers SKI algorithms (a.k.a. <b>injectors</b>) along with quality of service metrics (<b>QoS</b>).

[Here](https://link.springer.com/chapter/10.1007/978-3-031-15565-9_6) you can have a look at the original paper for more details.

If you use one or more of the features provided by PSyKI, please consider citing this work.

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

## Overview

Premise: the knowledge that we consider is symbolic, and it is represented with formal logic.
In particular, we use the Prolog formalism to express logic rules.

Note: some aspects of the <b>Prolog</b> language are not fully supported.
Generally, every SKI method specifies which kind of knowledge can support.

### SKI workflow

SKI methods require common steps for knowledge preprocessing.
First, the knowledge is parsed into a visitable data structure (e.g., abstract syntax tree).
Then, it is fuzzified.
This means that from a <i>crispy</i> domain -- logic rules can be only true or false -- the knowledge becomes <i>fuzzy</i> --- there can be multiple degree of truth.
Finally, the knowledge can be injected into the neural network.

In the literature, there are mainly two families of SKI methods:
- <b>structuring</b>, the knowledge is mapped into new neurons and connections of the neural network.
The new components mimic the behaviour of the prior knowledge.
After the injection, the network is still trained (knowledge fitting).
![SKI structuring](injection-structuring.png)
- <b>constraining</b>, the knowledge is embedded in the loss function.
Typically, a cost factor is added to the loss function.
The cost factor is higher the higher is the violation of the knowledge.
In this way, the network learn to avoid predictions that violates the prior knowledge during the training phase.
![SKI constraining](injection-constraining.png)

### Architecture


## Features
### Injectors

### QoS


## Collaborators
