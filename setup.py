from setuptools import setup, find_packages

setup(
    name='cascade',
    version='0.0.1',
    install_requires=[
        'torch', "numpy",
        'importlib-metadata; python_version == "3.9"',
    ],
    packages=find_packages(include = ["cascade.*"], exclude=["*tests*"])
)