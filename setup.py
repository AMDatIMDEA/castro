#!/usr/bin/env python

import os.path
import os
from setuptools import dist, find_packages, setup


readme = ''

here = os.path.abspath(os.path.dirname(__file__))
readme_path = os.path.join(here, 'README.md')
if os.path.exists(readme_path):
    with open(readme_path, 'rb') as stream:
        readme = stream.read().decode('utf8')

setup(
    long_description=readme,
    name='castro',
    version='1.0',
    description='CASTRO - Constrained sequential Latin hypercube (with multidimensional uniformity) sampling',
    python_requires='>=3.9',
    project_urls={
        "repository": "https://github.com/AMDatIMDEA/castro"},
    author='Christina Schenk',
    author_email='christina.schenk@imdea.org',
    maintainer='Christina Schenk',
    license='GPL V3',
    keywords='Constrained sequential Latin hypercube (with multidimensional uniformity) sampling',
    packages=find_packages(include=['castro', 'castro.*']),
    install_requires=requirements,
    extras_require={
        'tests': ['pytest', 'codecov', 'pytest-cov'],
        'docs': ['sphinx', 'sphinx-rtd-theme', 'myst-parser', 'myst-nb', 'sphinx-panels', 'autodocs']
    }
)
