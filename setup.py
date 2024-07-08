# -*- coding: utf-8 -*-

try:
    from setuptools import setup, Extension
except ImportError:
    from distutils.core import setup

import os.path
import os
from setuptools import dist


readme = ''
here = os.path.abspath(os.path.dirname(__file__))
readme_path = os.path.join(here, 'README.md')
if os.path.exists(readme_path):
    with open(readme_path, 'rb') as stream:
        readme = stream.read().decode('utf8')

with open('requirements.txt') as f:
    required = f.read().splitlines()

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
    install_requires=required,
    #extras_require={"dev": ["pytest==5.*,>=5.2.0"]},

)
