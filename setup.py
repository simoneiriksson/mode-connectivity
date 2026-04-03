from setuptools import setup, find_packages

setup(
    name="modeconnectivity",                 # distribution name (pip install mypkg)
    version="0.0.0",
    description="Implementation of mode connectivity result",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    package_dir={"": "modeconnectivity"},
    packages=find_packages(where="modeconnectivity"),
    python_requires=">=3.10",
    install_requires=[
        "matplotlib==3.10.8",
        "numpy==2.4.3",
        "torch==2.10.0",
        "torchmetrics==1.8.2",
        "torchvision==0.25.0",
        "torchviz==0.0.3",
        "jupyter"
    ],
)