# simple setup.py
from setuptools import setup, find_packages

setup(
    name='fast-llm-trainer',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
    ],
)
