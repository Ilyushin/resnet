# pylint: disable=line-too-long, invalid-name, missing-docstring

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="resnet_models",
    version="1.0.1",
    author="Eugene Ilyushin",
    author_email="eugene.ilyushin@gmail.com",
    description="The package contains ResNet architectures which were developed on TensorFlow",
    long_description="The package contains ResNet architectures which were developed on TensorFlow",
    long_description_content_type="text/markdown",
    url="https://github.com/Ilyushin/resnet",
    packages=setuptools.find_packages(),
    package_dir={
        'resnet_models': 'src',
    },
    entry_points={
        'console_scripts': [
            'resnet_models=src.main:main',
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'tensorflow',
        'pylint',
        'signal-transformation'
    ],
)
