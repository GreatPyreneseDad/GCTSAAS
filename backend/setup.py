from setuptools import setup, find_packages

setup(
    name="gct-saas",
    version="0.1.0",
    description="Grounded Coherence Theory SaaS Platform",
    packages=find_packages(),
    install_requires=[
        "fastapi>=0.104.1",
        "uvicorn[standard]>=0.24.0",
        "pydantic>=2.5.0",
        "numpy>=1.24.3",
        "scipy>=1.11.4",
        "pandas>=2.1.4",
        "pytest>=7.4.3",
        "pytest-asyncio>=0.21.1",
        "pytest-cov>=4.1.0",
    ],
    python_requires=">=3.9",
    extras_require={
        "dev": [
            "black",
            "flake8",
            "mypy",
            "pre-commit",
        ]
    },
)
