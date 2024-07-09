#!/usr/bin/env python

from os import path
import os
import setuptools
from setuptools import dist, find_packages, setup
from setuptools.command.install import install
import subprocess
import sys

#import versioneer

# Workaround for the `use_2to3` issue
def remove_use_2to3():
    orig_setup = setuptools.setup

    def patched_setup(**kwargs):
        kwargs.pop('use_2to3', None)
        return orig_setup(**kwargs)

    setuptools.setup = patched_setup

remove_use_2to3()

# Post-installation for installation mode
class CustomInstallCommand(install):
    """Customized setuptools install command - uses pip with --no-cache-dir to install requirements."""
    def run(self):
        # Ensure we install the requirements with --no-cache-dir
        here = path.abspath(path.dirname(__file__))
        requirements_path = path.join(here, 'requirements.txt')
        if os.path.exists(requirements_path):
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--no-cache-dir', '-r', requirements_path])

        # Register the Jupyter kernel
        subprocess.check_call([sys.executable, '-m', 'ipykernel', 'install', '--user', '--name', 'castro_env', '--display-name', 'Python (castro_env)'])

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
    install_requires=requirements + [
        'jupyterlab',
        'ipykernel',
        'jupyter_contrib_nbextensions'
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
        'python_version == "3.9"': [
            'ipython>8.13, <8.19',
        ],
        'python_version >= "3.10"': [
            'ipython>8.19',
        ],
    },
    setup_requires=['setuptools<58.0.0'],
    cmdclass={
        'install': CustomInstallCommand,
    },

)
