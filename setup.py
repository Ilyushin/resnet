# pylint: disable=line-too-long, invalid-name, missing-docstring

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as fh:
    requirements = fh.readlines()

setuptools.setup(
    name="resnet_models",
    version="1.1.3",
    author="Eugene Ilyushin",
    author_email="eugene.ilyushin@gmail.com",
    description="The package contains ResNet architectures which were developed on TensorFlow",
    long_description="The package contains ResNet architectures which were developed on TensorFlow",
    long_description_content_type="text/markdown",
    url="https://github.com/Ilyushin/resnet",
    packages=setuptools.find_packages(),
    package_dir={
        'resnet_models': 'resnet_models',
    },
    entry_points={
        'console_scripts': [
            'resnet_models=resnet_models.main:main',
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=requirements
)
