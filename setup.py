#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Setup configuration for proyecto-python."""

from setuptools import setup, find_packages

setup(
    name="proyecto-python",
    version="0.1.0",
    description="Un proyecto Python con estructura profesional",
    author="Tu Nombre",
    author_email="tu.email@example.com",
    url="https://github.com/usuario/proyecto-python",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=23.0",
            "flake8>=6.0",
            "isort>=5.0",
            "mypy>=1.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
