#!/usr/bin/env python

from os import path
import os
import setuptools
from setuptools import dist, find_packages, setup
from setuptools.command.install import install
import subprocess
import sys
import pkg_resources


def read_requirements():
    with open("requirements.txt") as req_file:
        return req_file.read().splitlines()



#with open('HISTORY.rst') as history_file:
#    history = history_file.read()

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
    #long_description=readme+'\n\n',
    packages=find_packages(include=['castro', 'castro.*']),
    install_requires=read_requirements(),
    url="https://gitlab.com/AMDatIMDEA/castro",
    version='1.1.0',
    classifiers=[
        "Programming Language :: Python :: 3.13",
    ],
)
