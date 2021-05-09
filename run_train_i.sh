#!/usr/bin/env bash


/bin/bash build_local.sh

pip3 install --upgrade signal-transformation

nohup resnet_models -t \
    --format stft \
    -a resnet_34 \
    --input-dev /home/disk/data/datasets/voice/vox1/dev/ \
    --input-eval /home/disk/data/datasets/voice/vox1/test/ \
    -o /home/disk/data/projects/resnet/test/ \
    --save-model /home/disk/data/projects/resnet/test/model/resnet_34/ \
    -b 30 \
    -e 100 &
