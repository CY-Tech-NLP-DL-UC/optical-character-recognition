# optical-character-recognition (ocrapi)

[![Build Status](https://travis-ci.org/CY-Tech-NLP-DL-UC/optical-character-recognition.svg?branch=main)](https://travis-ci.org/CY-Tech-NLP-DL-UC/optical-character-recognition)
[![codecov](https://codecov.io/gh/CY-Tech-NLP-DL-UC/optical-character-recognition/branch/main/graph/badge.svg)](https://codecov.io/gh/CY-Tech-NLP-DL-UC/optical-character-recognition)

## Table of contents
* [General info](#general-info)
* [Technologies](#technologies)
* [Project installation](#project-installation)
* [Starting the API](#starting-the-api)
* [Tests](#tests)
* [How to contribute](#how-to-contribute)

## General info

Small API that will have several features :

* Read license plate from pictures
* Read text from images. (letters...)

## Technologies

This project was created using:

* Python 3.6
* Flask 0.12.2
* Pytorch
  * `torch` 1.6.0
  * `torchvision` 0.7.0
 * Pillow 7.2.0

## Project installation

After cloning project, you may need to install the flask package.
To do so, follow these commands (do not ommit to use python environment if needed) :

```sh
# Move into the repository
cd optical-character-recognition
# Install the ocrapi package
pip install -e .
```

## Setting up the virtual environment

In order to avoid problems with versions, you can set up a virtual environment. Follow these steps to create and configure the virtual environment.

Creation :

```sh
python3 -m venv <path/to/your/virtual/env>
source bin/activate
```

Configuration :

```sh
pip3 install --upgrade pip
pip3 install opencv-python
pip3 install editdistance
pip3 install tensorflow
pip3 install matplotlib
pip3 install networkx
pip3 install imutils
pip3 install torch
pip3 install https://files.pythonhosted.org/packages/8c/52/33d739bcc547f22c522def535a8
da7e6e5a0f6b98594717f519b5cb1a4e1/torchvision-0.1.8-py2.py3-none-any.whl
pip3 install keras
```

Quit the virtual environment :

```sh
deactivate
```


## Starting the API:

In the root directory `ocrapi`:

```sh
python3 main.py
```

You should get an output in your terminal that looks like this:

```sh
 * Serving Flask app "main" (lazy loading)
 * Environment: development
 * Debug mode: on
 * Running on http://127.0.0.1:5001/ (Press CTRL+C to quit)
 * Restarting with stat
 * Debugger is active!
 * Debugger PIN: 237-165-773
```

## Tests

Unit tests are written in the `tests/` directory. You may run `pytest` from the root directory using the following command:

```sh
python -m pytest
```

## How to contribute

To ensure code quality, you should check that you have installed `pre-commit` in your `.git` folder using the command:

```sh
pre-commit install
```

You might need to install first the `pre-commit` package using the command:

```sh
pip install pre-commit
```
