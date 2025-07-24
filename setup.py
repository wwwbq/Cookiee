from setuptools import setup, find_packages

setup(
    name="cookiee",
    version="0.1",
    entry_points={
        "console_scripts": [
            "cookiee=cookiee.run:main",
        ],
    },
    install_requires=[
        "torch",
        "torchvision",
        "transformers",
        "datasets",
        "peft",
        "addict",
        "yapf",
        "helper @ git+https://github.com/wwwbq/helper.git",
    ],
    packages=find_packages(),
)