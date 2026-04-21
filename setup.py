from setuptools import setup, find_packages

setup(
    name="cookiee",
    version="0.1",
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'cookiee=cookiee.run:main'
        ]
    },
    install_requires=[
        "torch",
        "torchvision",
        "transformers==4.56.0",
        "datasets",
        "peft",
        "helper @ git+https://github.com/wwwbq/helper.git",
        "addict",
        "yapf"
        # pip install flash-attn==2.8.2 --no-build-isolation
        # pip install deepspeed
    ],
)