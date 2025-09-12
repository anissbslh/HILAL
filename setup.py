from setuptools import setup, find_packages
import os

cwd = os.path.dirname(os.path.abspath(__file__))
version_path = os.path.join(cwd, 'HILAL', '__version__.py')
with open(version_path) as fh:
    version = fh.readlines()[-1].split()[-1].strip("\"'")

with open("README.md", "r") as f:
    long_description = f.read()

requirements = []
with open("requirements.txt", "r") as f:
    for line in f:
        requirements.append(line.strip())

setup(
    name="HILAL",
    version=version,
    description="Hessian-Informed Layer Allocation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='aniss',
    author_email='ka_bessalah@esi.dz',
    license='Apache 2.0',
    keywords=['analog in-memory computing', 'heterogeneous systems', 'HILAL'],
    packages=find_packages(),
    python_requires='>=3.7',
    install_requires=requirements,
)