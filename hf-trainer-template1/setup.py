# setup.py
from setuptools import setup, find_packages

setup(
    name='hftrainer',
    version='0.0.1',
    packages=find_packages(),
    install_requires=[
        'transformers',
        'torch',
        'tqdm'
    ]
)