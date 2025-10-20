"""Setup configuration for PLM4NewsRS."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8")

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
with open(requirements_path) as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith('#'):
            requirements.append(line)

# Development dependencies
dev_requirements = [
    "pytest>=7.3.0",
    "pytest-cov>=4.1.0",
    "black>=23.3.0",
    "isort>=5.12.0",
    "flake8>=6.0.0",
    "pylint>=2.17.0",
    "mypy>=1.3.0",
    "pre-commit>=3.3.0",
]

# Documentation dependencies
docs_requirements = [
    "sphinx>=6.2.0",
    "sphinx-rtd-theme>=1.2.0",
    "sphinx-autodoc-typehints>=1.23.0",
]

setup(
    name="plm4newsrs",
    version="1.0.0",
    author="Thamolwan Poopradubsil",
    author_email="tmw.poopradubsil@gmail.com",
    description="Multi-architecture news recommendation with pre-trained language models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/thamolwanpo/plm4newsrs",
    project_urls={
        "Bug Tracker": "https://github.com/thamolwanpo/plm4newsrs/issues",
        "Documentation": "https://plm4newsrs.readthedocs.io",
        "Source Code": "https://github.com/thamolwanpo/plm4newsrs",
    },
    packages=find_packages(exclude=["tests", "tests.*", "notebooks", "scripts"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
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
        "dev": dev_requirements,
        "docs": docs_requirements,
        "all": dev_requirements + docs_requirements,
    },
    entry_points={
        "console_scripts": [
            "plm4newsrs-train=scripts.train:main",
            "plm4newsrs-evaluate=scripts.evaluate:main",
            "plm4newsrs-compare=scripts.compare_models:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)