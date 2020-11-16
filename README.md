# optical-character-recognition (ocrapi)

[![Build Status](https://travis-ci.org/CY-Tech-NLP-DL-UC/optical-character-recognition.svg?branch=main)](https://travis-ci.org/CY-Tech-NLP-DL-UC/optical-character-recognition)
[![codecov](https://codecov.io/gh/CY-Tech-NLP-DL-UC/optical-character-recognition/branch/main/graph/badge.svg)](https://codecov.io/gh/CY-Tech-NLP-DL-UC/optical-character-recognition)

Small API that will have several features :

* Read license plate from pictures
* Read text from images. (letters...)

## Project installation

After cloning project, you may need to install the flask package.
To do so, follow these commands (do not ommit to use python environment if needed) :

```sh
# Move into the repository
cd optical-character-recognition
# Install the ocrapi package
pip install -e .
```

## Starting the API:

In the root directory `ocrapi`:

```sh
flask run
```

You should get an output in your terminal that looks like this:

```sh
 * Serving Flask app "ocrapi" (lazy loading)
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
