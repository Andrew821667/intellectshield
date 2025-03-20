from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="intellectshield",
    version="0.1.0",
    author="Andrew821667",
    author_email="author@example.com",
    description="Библиотека для обнаружения аномалий и кибератак с использованием методов машинного обучения",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Andrew821667/intellectshield",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "Topic :: Security",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    test_suite="tests",
)
