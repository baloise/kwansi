from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="kwansi",
    version="0.1.0",
    author="Timm Süss",
    author_email="catchall@sporez.com",
    description="A library for creating and optimizing prompts for language models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lordamp/kwansi",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "dspy-ai",
        "python-dotenv",
    ],
    include_package_data=True,
    package_data={
        "kwansi": ["docs/*"],
    },
)