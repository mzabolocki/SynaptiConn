
from setuptools import setup, find_packages

def parse_requirements(filename):
    """Read dependencies from a requirements file."""
    with open(filename) as requirements_file:
        install_requires = requirements_file.read().splitlines()
        return install_requires

def long_description():
    with open("README.rst", "r", encoding="utf-8") as f:
        return f.read()

setup(
    name="synapticonn",
    version="0.0.1rc4",
    description="Inferring monosynaptic connections in neural circuits.",
    long_description=long_description(),
    long_description_content_type="text/x-rst",
    author="Michael Zabolocki",
    author_email="mzabolocki@gmail.com",
    maintainer_email="mzabolocki@gmail.com",
    url="https://github.com/mzabolocki/SynaptiConn",
    license='Apache License, 2.0',
    classifiers = [
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: MacOS',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    platforms='any',
    keywords=['neuroscience', 'connectomics', 'synaptic connections'],
    packages=find_packages(),
    install_requires=parse_requirements("requirements.txt"),
    test_suite='tests',
    tests_require=parse_requirements("requirements_dev.txt"),
    python_requires=">=3.8",
    include_package_data=True,
    zip_safe=False,
)