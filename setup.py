from setuptools import setup, find_packages

def parse_requirements(filename):
    """Read dependencies from a requirements file."""
    with open(filename, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

if __name__ == "__main__":
    setup(
        name="synapticonn",
        version="0.0.1",
        description="Inferring monosynaptic connections in neural circuits.",
        long_description=open("README.rst", "r", encoding="utf-8").read(),
        long_description_content_type="text/x-rst",
        author="Michael Zabolocki",
        author_email="mzabolocki@gmail.com",
        maintainer_email="mzabolocki@gmail.com",
        url="https://github.com/mzabolocki/SynaptiConn",
        packages=find_packages(),
        install_requires=parse_requirements("requirements.txt"),
        extras_require={
            "testing": parse_requirements("requirements_dev.txt"),
        },
        python_requires=">=3.8",
        include_package_data=True,
        zip_safe=False,
    )