from setuptools import setup

setup(
    name='bookbinder',
    version='0.1.0',
    packages=['bookbinder'],
    scripts=['bin/bookbinder'],
    install_requires=[
        'numpy'
    ],
)
