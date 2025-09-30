#!/usr/bin/env python3
"""
Setup script for Transformer Memory and FLOPS Calculator
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    requirements = []
    if os.path.exists("requirements.txt"):
        with open("requirements.txt", "r", encoding="utf-8") as fh:
            requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]
    return requirements

setup(
    name="transformer-calculator",
    version="1.0.0",
    author="Rakesh Cheerla",
    author_email="rakeshcheerla@example.com",
    description="Comprehensive transformer model memory and FLOPS calculator with MoE support",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/reunicles/Model_Calculator",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "web": ["flask>=2.0.0"],
        "dev": ["mypy>=1.0.0", "pytest>=7.0.0"],
        "all": ["flask>=2.0.0", "mypy>=1.0.0", "pytest>=7.0.0"],
    },
    entry_points={
        "console_scripts": [
            "transformer-calculator=src.cli_calculator:main",
            "transformer-web=src.web_interface_enhanced:main",
        ],
    },
    include_package_data=True,
    package_data={
        "src": ["templates/*.html", "static/*"],
    },
    keywords="transformer, memory, flops, calculator, moe, mixture-of-experts, flash-attention, gpu, ai, machine-learning",
    project_urls={
        "Bug Reports": "https://github.com/reunicles/Model_Calculator/issues",
        "Source": "https://github.com/reunicles/Model_Calculator",
        "Documentation": "https://github.com/reunicles/Model_Calculator/tree/main/docs",
    },
)


