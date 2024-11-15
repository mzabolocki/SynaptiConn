from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    long_description = readme_file.read()

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="synapticonn",
    version="0.0.1",
    description="Inferring monosynaptic connections in neural circuits.",
    author="Michael Zabolocki",
    author_email='mzabolocki@gmail.com',
    maintainer_email='mzabolocki@gmail.com',
    url='https://github.com/mzabolocki/SynaptiConn',
    long_description=long_description,
    packages=find_packages(),
    install_requires=requirements,
)