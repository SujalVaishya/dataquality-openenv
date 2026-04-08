from setuptools import setup, find_packages

setup(
    name="dataquality-env",
    version="1.0.0",
    description="OpenEnv-compliant data quality triage environment for AI agents",
    author="dataquality-env",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "pydantic>=2.0",
        "fastapi>=0.115",
        "uvicorn[standard]>=0.30",
        "openai>=1.40",
    ],
    extras_require={
        "dev": ["pytest>=8.0"],
    },
    entry_points={
        "console_scripts": [
            "dataquality-baseline=baseline.run_baseline:main",
        ],
    },
)
