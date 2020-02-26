#!/usr/bin/env bash


rm -rf ./resnet_models.egg-info ./dist ./build
python3 setup.py sdist bdist_wheel
pip3 uninstall -y resnet-models
pip3 install ./dist/resnet_models-*-py3-none-any.whl