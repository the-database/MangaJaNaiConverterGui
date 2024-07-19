from setuptools import setup
from Cython.Build import cythonize

setup(
    name='test run auto levels folder',
    ext_modules=cythonize("./backend/src/testrunautolevelszipmulti.py"),
)
