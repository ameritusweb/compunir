from setuptools import setup, find_packages

setup(
    name="compunir",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=2.0.0",
        "grpcio>=1.50.0",
        "grpcio-tools>=1.50.0",
        "aiohttp>=3.8.0",
        "pynvml>=11.0.0",
        "PyYAML>=6.0",
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "cryptography>=3.4.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.20.0",
            "black>=22.0.0",
            "isort>=5.0.0",
            "mypy>=0.900",
            "pytest-cov>=3.0.0",
        ]
    },
    python_requires=">=3.8",
    author="ameritusweb",
    author_email="ameritusweb@gmail.com",
    description="Decentralized GPU training network with verification and payment systems",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ameritusweb/compunir",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)