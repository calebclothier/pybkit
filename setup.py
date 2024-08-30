from setuptools import setup, find_packages

setup(
    name='pybkit',
    version='0.1',
    packages=find_packages(),
    description='Software representation of a neutral atom quantum computer',
    long_description=open('README.md').read(),
    author='Caleb Clothier',
    install_requires=[
        'numpy', 
        'pandas'
    ],
)
