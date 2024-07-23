# <img alt="CASTRO" src="branding/CASTRO-Logo.svg" height="80">

[![](https://img.shields.io/github/license/AMDatIMDEA/castro)](https://github.com/AMDatIMDEA/castro/blob/master/LICENSE)
[![](https://img.shields.io/github/last-commit/AMDatIMDEA/castro)](https://github.com/AMDatIMDEA/castro/)
[![Documentation Status](https://readthedocs.org/projects/castro/badge/?version=latest)](https://castro.readthedocs.io/en/latest/py-modindex.html)


CASTRO is a Python package designed to ...

- **Documentation:** - https://castro.readthedocs.io
- **Examples and Tutorials** - https://github.com/AMDatIMDEA/castro/examples
- **Source code:** - https://github.com/AMDatIMDEA/castro/src
- **Bug reports:** - https://github.com/AMDatIMDEA/castro/issues

It has the following functionality:
 - sampling with space coverage for mixture and other synthesis constraints
 - uses divide and conquer approach for problem of dimension greater than 4, divide problem into subproblems and later reassemble
 - find n_des experiments to conduct for exploration of the design space under a limited budget and taking the previously collected experimental data into account
 - pre- and post-process data
 - calculation of uniformity metrics
 - visualization of results

<br>

## Installation

You can install CASTRO by simply cloning or downloading the repository, installing the packages in the requirements.txt file, and then installing CASTRO. First, create a virtual environment called castro_env via:
    virtualenv castro_env
Then activate the virtual environment by:
    source castro_env/bin/activate
Next you can install castro in this virtual environment:
    cd <installation directory>
    git clone https://github.com/AMDatIMDEA/castro.git
    cd castro
    python setup.py install

Note: if it does not succeed the first time, try to rerun:
    python setup.py install


### Examples and Tutorials

The example problems are in the examples subfolder at https://github.com/AMDatIMDEA/castro/examples.
The data in the data folder was provided by José Hobson and De-Yi Wang (IMDEA Materials Institute).
<br>

## License

GPL-3


## Authors

    - Christina Schenk - IMDEA Materials Institute

## Please cite
<br>

 - C. Schenk, M. Haranczyk (2024): CASTRO - Efficient constrained sampling method for material and chemical experimental design, ArXiV, (under review)
