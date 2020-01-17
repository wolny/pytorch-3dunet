from setuptools import setup, find_packages

_version_ = "1.0.2"

setup(
    name="pytorch3dunet",
    packages=find_packages(exclude=["tests"]),
    version=_version_,
    author="Adrian Wolny, Lorenzo Cerrone",
    url="https://github.com/wolny/pytorch-3dunet",
    license="MIT",
    python_requires='>=3.7'
)
