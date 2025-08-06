from setuptools import setup, find_packages

setup(
    name="photonic_ai_simulator",
    version="0.1.0",
    description="Optical computing simulation framework for 100x ML acceleration",
    author="Daniel Schmidt",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        # Core dependencies will be added based on research needs
    ],
    python_requires=">=3.8",
)
