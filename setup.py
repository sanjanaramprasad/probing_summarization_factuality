from setuptools import setup, find_packages
from codecs import open
from os import path


with open(path.join(path.abspath(path.dirname(__file__)), 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup_info = dict(
     name='probing_summarization_factuality',
     version='1.0.0',
     author='Sanjana Ramprasad',
     author_email='ramprasad.sa@northeastern.edu',
     description='probing_summarization_factuality',
     long_description=long_description,
     packages=find_packages()
)

setup(**setup_info)