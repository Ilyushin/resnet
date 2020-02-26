# ResNet models

This library contains ResNet models, such as ResNet 34, ResNet 50, and functionality for helping train them on VoxCeleb1 for the speaker recognition task.

### Installation:

```bash
chmod +x ./build_local.sh
./build_local.sh
```

### Execution:
```bash
resnet_models -t \
    -a resnet_34 \ 
    --input-dev ./vox1/dev/wav/ \
    --input-eval ./vox1/test/wav/ \
    -o ./data/ \
    --save-model ./test/model/ \
    -b 300
```