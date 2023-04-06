# Name-Entity-Recognition

Created two models to slove the Name entity recognition. Name-Entity-Recognition (NER) is a subtask of Natural Language Processing (NLP) that involves identifying and extracting entities such as person names, organizations, locations, and other types of named entities from unstructured text.


1. [General](#General)
3. [Program Structure](#Program-Structure)
    - [Network Structure](#Network-Structure)
5. [Installation](#Installation)

## General
The goal is to create model that will recogize the right entity of each word in the sentence.



## Program Structure
* model1_318170917_3222995358.py - the first model, SVC
* model23_318170917_3222995358.py - the second model, fully connected network.
* generate_comp_tagged.py - uses the model to tag the data.

### Network-Structure
The network is fc based network including three layers with the following dimensions

<img src="https://i.imgur.com/a2HMprY.png" width = 50% height=50%>
 

## Installation
1. Open the terminal

2. Clone the project by:
```
    $ git clone https://github.com/elaysason/Name-Entity-Recognition.git
```
3. Run the generate_comp_tagged.py.py file by:
```
    $ python generate_comp_tagged.py.py
```
