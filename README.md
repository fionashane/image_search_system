# INFS 2044 SP5 2023 Assignment 2

The code given in this folder provides a skeleton and some functions that you can use to develop your assignment 2 solution.

## Installation

Several packages are required for this code to work. They are listed in file `requirements.txt`.

It is recommended that you create a dedicated Python environment for this assignment. That is, do not install the necessary packages in your system-wide Python environment.

To create a Python environment, you can use [conda](https://www.anaconda.com/download) or venv/pip.

Using `conda` (run from a terminal/command line):

```shell
conda create --channel conda-forge --name infs2044a2 --file requirements.txt

conda activate infs2044a2
```

Instructions for `venv` and `pip` are at <https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/>

If you use an IDE such as [PyCharm](https://www.jetbrains.com/pycharm/) or [Visual Studio Code](https://code.visualstudio.com/), you will need to select this environment as the Python interpreter.

## Execution

You can run the code from a terminal/command line using the example commands below:

```shell
python image_search.py search --all car person

python image_search.py similar --k 3 example_images/image3.jpg

python image_search.py similar example_images/image7.jpg 

python image_search.py list
```
