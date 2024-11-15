from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="synapticonn",
    version="0.0.1",
    description="Inferring monosynaptic connections in neural circuits",
    author="Michael Zabolocki",
    packages=find_packages(),
    install_requires=requirements,
)