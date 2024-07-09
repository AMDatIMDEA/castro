#!/usr/bin/env python

from os import path
import os
import setuptools
from setuptools import dist, find_packages, setup
from setuptools.command.install import install
import subprocess
import sys
import pkg_resources

#import versioneer

# Workaround for the `use_2to3` issue
def remove_use_2to3():
    orig_setup = setuptools.setup

    def patched_setup(**kwargs):
        kwargs.pop('use_2to3', None)
        return orig_setup(**kwargs)

    setuptools.setup = patched_setup

remove_use_2to3


# Function to determine the appropriate IPython version
def get_ipython_dependency():
    if sys.version_info >= (3, 10):
        return "ipython>=8.19"
    elif sys.version_info >= (3, 9):
        return "ipython>=8.13,<8.19"
    else:
        raise RuntimeError("Python version not supported by IPython")


# Post-installation for installation mode
class CustomInstallCommand(install):
    """Customized setuptools install command - uses pip with --no-cache-dir to install requirements."""
    def run(self):
        # Ensure we install IPython with the correct version requirement
        ipython_dependency = get_ipython_dependency()
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--no-cache-dir', ipython_dependency])

        # Ensure ipykernel is installed
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--no-cache-dir', 'ipykernel'])

        # Register the Jupyter kernel
        try:
            subprocess.check_call([sys.executable, '-m', 'ipykernel', 'install', '--user', '--name', 'castro_env', '--display-name', 'Python (castro_env)'])
        except subprocess.CalledProcessError as e:
            print(f"Failed to register Jupyter kernel: {e}")
            sys.exit(1)


        # Run the standard install command
        install.run(self)

with open('README.md') as readme_file:
    readme = readme_file.read()

#with open('HISTORY.rst') as history_file:
#    history = history_file.read()


here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'requirements.txt')) as requirements_file:
    # Parse requirements.txt, ignoring any commented-out lines.
    requirements = [line for line in requirements_file.read().splitlines()
                    if not line.startswith('#')]

setup(
    name='castro',
    description='CASTRO is a code for a novel constrained sequential Latin hypercube (with multidimensional uniformity) sampling method',
    python_requires='>=3.9',
    project_urls={
        "repository": "https://github.com/AMDatIMDEA/castro"},
    author='Christina Schenk',
    author_email='christina.schenk@imdea.org',
    maintainer='Christina Schenk',
    license='GPL V3',
    keywords='castro',
    long_description=readme+'\n\n',
    packages=find_packages(include=['castro', 'castro.*']),
    version='1.0.0',
    #version=versioneer.get_version(),
    #cmdclass=versioneer.get_cmdclass(),
    install_requires = requirements + [
        'jupyterlab',
        'ipykernel',
        'jupyter_contrib_nbextensions'
    ] + ([
        'ipython>=8.19'  # Common dependency for all Python versions >= 3.10
    ] if sys.version_info >= (3, 10) else [
        'ipython>=8.13, <8.19'  # Dependency specific to Python 3.9
    ]),
    extras_require={
        'tests': ['pytest', 'codecov', 'pytest-cov'],
        'docs': [
            'sphinx==7.3.7',
            'myst-parser',
            'myst-nb',
            'sphinx-panels',
            'sphinxcontrib.devhelp==1.0.5',
            'sphinxcontrib.applehelp==1.0.2',
            'sphinxcontrib.htmlhelp==2.0.4',
            'sphinxcontrib.serializinghtml==1.1.9',
            'sphinxcontrib.qthelp==1.0.6',
            'docutils<0.21',
            'sphinx-rtd-theme',
            'sphinxcontrib.viewcode',
            'sphinx.ext.napoleon',
            'sphinxcontrib.mathjax',
            'sphinxcontrib.jquery==4.0',
            'autodocs'
        ],
    },
    #setup_requires=['setuptools<58.0.0'],
    cmdclass={
        'install': CustomInstallCommand,
    },

)
