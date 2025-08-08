from setuptools import setup, find_packages

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

# Read long description
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="photonic_ai_simulator",
    version="0.1.0",
    author="Daniel Schmidt",
    author_email="daniel@terragonlabs.com",
    description="Optical computing simulation framework for 100x ML acceleration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/danieleschmidt/photonic-ai-simulator",
    project_urls={
        "Bug Reports": "https://github.com/danieleschmidt/photonic-ai-simulator/issues",
        "Source": "https://github.com/danieleschmidt/photonic-ai-simulator",
        "Documentation": "https://photonic-ai-simulator.readthedocs.io",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "gpu": [
            "cupy-cuda11x>=10.0.0",
            "nvidia-ml-py3>=7.352.0",
        ],
        "jax": [
            "jax>=0.4.0",
            "jaxlib>=0.4.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "pytest-benchmark>=4.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "pre-commit>=2.20.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "sphinx-autodoc-typehints>=1.12.0",
            "myst-parser>=0.18.0",
        ],
        "jupyter": [
            "jupyter>=1.0.0",
            "jupyterlab>=3.0.0",
            "ipywidgets>=7.6.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "photonic-benchmark=photonic_ai_simulator.cli:benchmark_command",
            "photonic-train=photonic_ai_simulator.cli:train_command",
            "photonic-validate=photonic_ai_simulator.cli:validate_command",
        ],
    },
    include_package_data=True,
    package_data={
        "photonic_ai_simulator": [
            "data/*.json",
            "configs/*.yaml",
            "models/*.pkl",
        ],
    },
    zip_safe=False,
    keywords=[
        "photonic computing",
        "optical neural networks", 
        "machine learning acceleration",
        "quantum computing",
        "neuromorphic computing",
        "hardware acceleration",
        "deep learning",
        "photonics",
        "optics",
        "simulation",
    ],
)