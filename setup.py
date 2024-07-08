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

class PostInstallCommand(install):
    """Post-installation for installation mode."""
    def run(self):
        install.run(self)
        # Register the kernel after install
        subprocess.check_call([sys.executable, '-m', 'castro.post_install'])

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
    version='1.0',
    #version=versioneer.get_version(),
    #cmdclass=versioneer.get_cmdclass(),
    install_requires=requirements + [
        'jupyterlab',
        'ipykernel'
    ],,
    extras_require={
        'tests': ['pytest', 'codecov', 'pytest-cov'],
        'docs': ['sphinx', 'sphinx-rtd-theme', 'myst-parser', 'myst-nb', 'sphinx-panels', 'autodocs']
    },
    setup_requires=['setuptools<58.0.0'],
    entry_points={
        'console_scripts': [
            'castro-jupyter = castro.jupyter:main',
        ],
    },
)
