#!/usr/bin/env bash


rm -rf ./resnet_models.egg-info ./dist ./build
python3 setup.py sdist bdist_wheel
python3 -m twine upload dist/*