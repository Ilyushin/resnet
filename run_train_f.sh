#!/usr/bin/env bash


/bin/bash build_local.sh

pip3 install --upgrade signal-transformation

nohup resnet_models -t \
    --format stft \
    -a resnet_34 \
    --input-dev /home/datasets/voice/vox1/dev/ \
    --input-eval /home/datasets/voice/vox1/test/ \
    -o /home/projects/resnet/test/data/ \
    --save-model /home/projects/resnet/test/model/resnet_34/ \
    -b 20 \
    -e 100 &
