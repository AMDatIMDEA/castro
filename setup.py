#!/usr/bin/env python

import os
import subprocess
import sys
from os import path
from setuptools import setup, find_packages
from setuptools.command.install import install
import pkg_resources

# Workaround for the `use_2to3` issue
def remove_use_2to3():
    orig_setup = setup

    def patched_setup(**kwargs):
        kwargs.pop('use_2to3', None)
        return orig_setup(**kwargs)

    setup = patched_setup

remove_use_2to3()

# Function to determine the appropriate IPython version
def get_ipython_dependency():
    python_version = pkg_resources.parse_version(sys.version.split(" ")[0])
    if python_version < pkg_resources.parse_version("3.10"):
        return "ipython>=8.13,<8.19"
    else:
        return "ipython>=8.19"

# Post-installation for installation mode
class CustomInstallCommand(install):
    """Customized setuptools install command - uses pip with --no-cache-dir to install requirements."""
    def run(self):
        # Ensure we install the requirements with --no-cache-dir
        here = path.abspath(path.dirname(__file__))
        requirements_path = path.join(here, 'requirements.txt')
        if os.path.exists(requirements_path):
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--no-cache-dir', '-r', requirements_path])

        # Ensure ipykernel is installed
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--no-cache-dir', 'ipykernel'])
        except subprocess.CalledProcessError as e:
            print(f"Failed to install ipykernel: {e}")
            sys.exit(1)

        # Register the Jupyter kernel
        try:
            subprocess.check_call([sys.executable, '-m', 'ipykernel', 'install', '--name', 'castro_env2', '--display-name', 'Python (castro_env2)'])
        except subprocess.CalledProcessError as e:
            print(f"Failed to register Jupyter kernel: {e}")
            print(f"Output: {e.output}")
            sys.exit(1)

        # Run the standard install command
        install.run(self)

with open('README.md') as readme_file:
    readme = readme_file.read()

here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'requirements.txt')) as requirements_file:
    requirements = [line for line in requirements_file.read().splitlines() if not line.startswith('#')]

setup(
    name='castro',
    version='1.0.0',
    description='CASTRO is a code for a novel constrained sequential Latin hypercube (with multidimensional uniformity) sampling method',
    python_requires='>=3.9',
    project_urls={
        "repository": "https://github.com/AMDatIMDEA/castro"
    },
    author='Christina Schenk',
    author_email='christina.schenk@imdea.org',
    maintainer='Christina Schenk',
    license='GPL V3',
    keywords='castro',
    long_description=readme,
    packages=find_packages(include=['castro', 'castro.*']),
    install_requires=requirements + [
        'jupyterlab',
        'ipykernel',
        'jupyter_contrib_nbextensions',
        get_ipython_dependency()
    ],
    extras_require={
        'tests': ['pytest', 'codecov', 'pytest-cov'],
        'docs': [
            'sphinx',
            'sphinx-rtd-theme',
            'myst-parser',
            'myst-nb',
            'sphinx-panels',
            'sphinxcontrib.applehelp',
            'sphinxcontrib.devhelp',
            'sphinxcontrib.htmlhelp',
            'sphinxcontrib.qthelp',
            'sphinxcontrib.serializinghtml',
            'autodocs'
        ],
    },
    setup_requires=['setuptools<58.0.0'],
    cmdclass={
        'install': CustomInstallCommand,
    },
)
