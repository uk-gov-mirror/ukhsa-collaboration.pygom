# Installation

Installation instructions may be found on the [GitHub project README](https://github.com/ukhsa-collaboration/pygom/), but we include them here also.

## From source

Source code for PyGOM can be downloaded from the GitHub repository: https://github.com/ukhsa-collaboration/pygom

Clone the repository and install from the project root:

```bash
git clone https://github.com/ukhsa-collaboration/pygom.git

cd pygom

python -m pip install .
```

For development, an editable install can be used

```bash
python -m pip install -e .
```

```{note}
Please be aware that there may be redundant files within the package as it is under active development.
The latest fully reviewed version of PyGOM will be on the master branch and we recommend that users install the version from there.
```

The final prerequisite, if you are working on a Windows machine, is that you will also need to install:
If you want the visualisation features, you need graphviz.
This feature may be phased out or a pure python implementation found
- [Graphviz](https://graphviz.org/)

## From PyPI

Alternatively, the latest release can be installed from [PyPI](https://pypi.org/project/pygom/):

```bash
pip install pygom
```

# Testing the package

Test files should then be run from the command line to check that installation has completed successfully

```bash
python setup.py test
```

This can take some minutes to complete.
