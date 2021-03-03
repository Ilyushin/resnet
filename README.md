# ResNet models

This library contains ResNet models, such as ResNet 34, ResNet 50, and functionality for helping 
train them on [VoxCeleb1](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html) for the speaker recognition task.
Some parts of the architecture were taken from http://www.robots.ox.ac.uk:5000/~vgg/publications/2019/Xie19a/xie19a.pdf
TensorFlow implementation - https://github.com/WeidiXie/VGG-Speaker-Recognition

### Installation:

```bash
chmod +x ./build_local.sh
./build_local.sh
```

### Execution:
```bash
resnet_models -t \
    -p \
    -a resnet_34 \ 
    --input-dev ./vox1/dev/wav/ \
    --input-eval ./vox1/tests/wav/ \
    -p \
    -o ./data/ \
    --save-models ./tests/models/ \
    -b 300
```

### Pre-trained models
1) [ResNet 34](https://1drv.ms/u/s!AgTCRsS6Et0Cvptmqy_NQ1EGM8rSiA?e=m1y8Cj)