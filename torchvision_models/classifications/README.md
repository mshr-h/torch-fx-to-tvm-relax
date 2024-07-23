# Image Classification Models from Torchvision

## Run

```bash
python classification_test.py --model MODEL_NAME

# example
python classification_test.py --model alexnet
```

## Supporting matrix

| Model                                                  | Status                                                                                                    |
| ------------------------------------------------------ | --------------------------------------------------------------------------------------------------------- |
| alexnet                                                | Supported ✅                                                                                               |
| convnext_{base,large,small,tiny}                       | missing support for [torch.permute #17184](https://github.com/apache/tvm/pull/17184) and stochastic_depth |
| densenet{121,161,169,201}                              | Supported ✅                                                                                               |
| efficientnet_b{0-7}                                    | Supported ✅                                                                                               |
| efficientnet_v2_{l,m,s}                                | Supported ✅                                                                                               |
| googlenet                                              | fails when constant folding                                                                               |
| inception_v3                                           | missing support for [max_pool2d #17189](https://github.com/apache/tvm/pull/17189)                         |
| maxvit_t                                               | missing support for [einsum #17186](https://github.com/apache/tvm/pull/17186) and stochastic_depth        |
| mnasnet{0_5,0_75,1_0,1_3}                              | Supported ✅                                                                                               |
| mobilenet_v2                                           | Supported ✅                                                                                               |
| mobilenet_v3_{large,small}                             | Supported ✅                                                                                               |
| regnet_x_{400mf,800mf,1_6gf,3_2gf,8gf,16gf,32gf}       | Supported ✅                                                                                               |
| regnet_y_{400mf,800mf,1_6gf,3_2gf,8gf,16gf,32gf,128gf} | Supported ✅                                                                                               |
| resnet{18,34,50,101,152}                               | Supported ✅                                                                                               |
| resnext{50_32x4d,101_32x8d,101_64x4d}                  | Supported ✅                                                                                               |
| shufflenet_v2_x{0_5,1_0,1_5,2_0}                       | Supported ✅                                                                                               |
| squeezenet{1_0,1_1}                                    | Supported ✅                                                                                               |
| swin_{b,s,t}                                           | missing support for _get_relative_position_bias                                                           |
| swin_v2_{b,s,t}                                        | missing support for _get_relative_position_bias                                                           |
| vgg{11,11_bn,13,13_bn,16,16_bn,19,19_bn}               | Supported ✅                                                                                               |
| vit_{b_16,b_32,h_14,l_16,l_32}                         | not yet                                                                                                   |
| wide_resnet{50_2,101_2}                                | Supported ✅                                                                                               |
