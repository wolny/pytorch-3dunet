from setuptools import setup, find_packages

setup(
    name="pytorch3dunet",
    packages=find_packages(exclude=["tests"]),
    version="1.0.2",
    author="Adrian Wolny, Lorenzo Cerrone",
    url="https://github.com/wolny/pytorch-3dunet",
    license="MIT",
    python_requires='>=3.7'
)
