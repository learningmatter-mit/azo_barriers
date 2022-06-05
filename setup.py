import os
import io

from setuptools import setup, find_packages


def read(fname):
    with io.open(os.path.join(os.path.dirname(__file__), fname), encoding="utf-8") as f:
        return f.read()


setup(
    name="azo_barriers",
    version="1.0.0",
    author="Simon Axelrod",
    email="simonaxelrod83@gmail.com",
    url="https://github.com/learningmatter-mit/azo_barriers",
    packages=find_packages("."),
    scripts=["scripts/get_barriers.py"],
    python_requires=">=3.8",
    install_requires=["numpy", "networkx", "pymatgen>=2020.3.2", "scikit-learn"],
    license="MIT",
    description="Package for computing thermal barriers of azobenzene derivatives",
    long_description=read("README.md"),
)
