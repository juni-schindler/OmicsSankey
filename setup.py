"""Setup."""

from setuptools import find_packages
from setuptools import setup

__version__ = "0.1.0"


setup(
    name="OmicsSankey",
    author="Bowen Tan"
    author_email= "bowentan3-c@my.cityu.edu.hk",
    version=__version__,
    install_requires=[
        "numpy",
    ],
    zip_safe=False,
    packages=find_packages(),
    include_package_data=True,
)