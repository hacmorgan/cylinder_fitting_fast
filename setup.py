#!/usr/bin/env python3


"""
Install cylinder fitting library
"""


import os
from setuptools import setup


setup(
    name="cylinder_fitting_fast",
    description="Vectorised cylinder fitting algorithm",
    # author="Abyss Solutions",
    # author_email="tech@abysssolutions.com.au",
    # url="http://github.com/abyss-solutions/abyss-bedrock",
    packages=["cylinder_fitting"],
    package_data={},
    install_requires=[
        "matplotlib",
        "numpy",
        "scipy",
    ],
    scripts=[],
    # zip_safe=False,
)
