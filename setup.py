"""
Setup for urdf_parser package
Author: Marco Magri and Carlo Nainer (Fraunhofer Italia)
"""


# from setuptools import setup, find_packages

# import os

# package_folder_path = os.path.dirname(os.path.realpath(__file__))
# requirements_path = os.path.join(package_folder_path, "requirements.txt")

# if os.path.isfile(requirements_path):
#     with open(requirements_path) as f:
#         requirements = f.read().splitlines()
#         print(requirements)


# with open("LICENSE", "r") as f:
#     license = f.read()

# setup(
#     name="urdf_parser",
#     version="0.0.1",
#     author="Marco Magri",
#     author_email="marco.magri@fraunhofer.it",
#     description="URDF parser package",
#     license=license,
#     packages=["urdf_parser"],
#     install_requires=requirements,
#     test_suite="nose.collector",
#     tests_require=["nose"],
# )


from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup
d = generate_distutils_setup(
    packages=['urdf_parser'],
    package_dir={'': '.'}
)
setup(**d)