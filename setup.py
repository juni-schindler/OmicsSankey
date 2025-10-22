"""Setup."""

from setuptools import find_packages
from setuptools import setup

__version__ = "0.2.0"


setup(
    name="OmicsSankey",
    author="Bowen Tan, Juni Schindler",
    author_email= "bowentan3-c@my.cityu.edu.hk, juni.schindler19@imperial.ac.uk",
    version=__version__,
    install_requires=[
        "numpy",
    ],
    zip_safe=False,
    packages=find_packages(),
    include_package_data=True,
)