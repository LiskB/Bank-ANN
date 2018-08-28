# Bank Customer Artificial Neural Network

This is an artificial neural network that uses Python to predict whether a customer will leave a bank based on multiple independent variables (some of which are credit score, age and gender). The neural network is trained on a set of data and is applied to other data to test its accuracy.

This challenge was part of the Udemy course "[Deep Learning A-Z](https://www.udemy.com/deeplearning/)".

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

[Python 3.6.6](https://www.python.org/downloads/) (As of 28/08/2018, Python 3.7.x **is not** compatible, due to Tensorflow compatibility issues)

[PIP 18.0 Package Management](https://pypi.org/project/pip/)

```
python get-pip.py
```

Either [Anaconda 4.5.10](https://conda.io/docs/user-guide/install/index.html) or the following libraries:
* [NumPy](http://www.numpy.org/)
* [Matplotlib](https://matplotlib.org/)
* [Pandas](https://pandas.pydata.org/)

[Theano](https://github.com/Theano/Theano)

```
pip install theano
```

[TensorFlow](https://www.tensorflow.org/)

```
pip install tensorflow
```

[Keras](https://keras.io/)

```
pip install keras
```

### Installing

Download and install all prerequisites from above.

Clone or download this repository to obtain the program and the test data.

```
git clone https://github.com/LiskB/Bank-ANN.git
```

Open the ann.py file to view the code.

## Running the tests

The code is divided into four parts, labelled in the code. These parts **must not** be run all at once due to the neural network being built in multiple different ways within the one file (the code can be run in separate parts by using the Spyder IDE included with Anaconda, or with another IDE with similar functionality). Part 1, Data Preproccessing, must always be run. Parts 2 and 3 is one way to build the neural network, and Part 4 is an improved way. Therefore, the code can be properly executed in two ways: running Part 1 and Part 4, **OR**  running Part 1, Part 2, and Part 3.

To view the results of the neural network, type the appropriate variable name into the console.

**NOTE:** Running Part 4 automatically tunes the neural network, and may take several hours, depending on processing power.

## Built With

* [Python 3.6.6](https://www.python.org/) - The language used
* Multiple libraries mentioned above

## Authors

**Branden Lisk**

## Acknowledgments

This project scenario and test data were provided by the Udemy course "[Deep Learning A-Z](https://www.udemy.com/deeplearning/)". This is a project for part of the course, so some of the code was created through video walkthrough tutorials.
