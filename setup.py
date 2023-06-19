""" Script to install the project """

import os

from setuptools import find_packages, setup

this_code_dirname = os.path.dirname(os.path.realpath(__file__))
with open(
    os.path.join(this_code_dirname, "README.md"), "r", encoding="utf-8"
) as file:
    long_description = file.read()


setup(
    name="bowlrelease",
    author="Davide Zambrano",
    author_email="d.zambrano@sportradar.com",
    url="https://github.com/DeepSportradar/cricket-bowl-release-challenge",
    description="A toolkit for the DeepSportradar Bowl Release challenge. An opportunity to publish at MMSports @ ACMMM and to win 2x $500.",
    long_description=long_description,
    python_requires=">=3.7",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy",
        "pycocotools",
        "scipy",
        "torch",
        "torchvision",
        "vidgear",
    ],
)
