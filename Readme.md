# <img alt="CASTRO" src="branding/CASTRO-Logo.svg" height="80">

[![](https://img.shields.io/github/license/AMDatIMDEA/castro)](https://github.com/AMDatIMDEA/castro/blob/master/LICENSE)
[![](https://img.shields.io/github/last-commit/AMDatIMDEA/castro)](https://github.com/AMDatIMDEA/castro/)
[![Documentation Status](https://readthedocs.org/projects/castro/badge/?version=latest)](https://castro.readthedocs.io/en/latest/py-modindex.html)


CASTRO (ConstrAined Sequential laTin hypeRcube sampling methOd) is a Python package designed to facilitate experimental design space exploration under mixture and synthesis constraints. It is especially useful when additional experiments must be planned within a limited budget or previously collected data should be exploited.

- **Documentation:** - https://castro.readthedocs.io
- **Examples and Tutorials** - https://github.com/AMDatIMDEA/castro/tree/main/examples
- **Source code:** - https://github.com/AMDatIMDEA/castro/tree/main/src
- **Bug reports:** - https://github.com/AMDatIMDEA/castro/issues

It has the following functionality:
 - Sampling with space coverage for mixture and other synthesis constraints.
 - Uses divide and conquer approach for problem of dimension greater than 4, divide problem into subproblems and then later reassemble the solutions.
 - Find limited number of desired experiments to conduct for exploration of the design space under a limited budget while taking previously collected experimental data into account.
 - Quantitative design quality metrics: central and warp-around discrepancy, and variance.
 - Data pre- and post-processing.
 - Visualization of results.

<br>

## Installation
Before installing CASTRO, you need to install git and a Python version (tested for 3.13).

You can install CASTRO by simply cloning or downloading the repository, installing the packages in the requirements.txt file, and then installing CASTRO. This was tested for python 3.13.

First, create a virtual environment called castro_env via:
    - virtualenv castro_env
or:
    - python3 -m venv castro_env
Then activate the virtual environment by:
    - source castro_env/bin/activate

or with Anaconda (Windows) in the Anaconda prompt to create a virtual environment:
- conda create -n myenv python=3.13
Then to activate it:
- conda activate myenv

- Next you can install castro in this virtual environment:
    - cd "installation_directory"
    - git clone https://github.com/AMDatIMDEA/castro.git
    - cd castro
    - pip install -e .



### Examples and Tutorials

The example problems are in the examples subfolder at https://github.com/AMDatIMDEA/castro/tree/main/examples.
The data in the data folder was provided by José Hobson and De-Yi Wang (IMDEA Materials Institute).
<br>

## License

GPL-3


## Authors

    - Christina Schenk - IMDEA Materials Institute

## Please cite
<br>

 - C. Schenk, M. Haranczyk (2025): A Novel Constrained Sampling Method for Efficient Exploration in Materials and Chemical Mixture Design, Computational Materials Science, 252:113780, DOI: https://doi.org/10.1016/j.commatsci.2025.113780
