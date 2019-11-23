#!/usr/bin/env python

from setuptools import setup, find_packages


setup(
    name='genetic_tournament_generator',
    version='0.1',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    include_package_data=True,
    install_requires=[
        'pandas',
    ],
)