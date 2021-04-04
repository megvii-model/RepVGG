# RepVGG: Making VGG-style ConvNets Great Again (CVPR-2021)

The official PyTorch implementation, pretrained models and examples are available at 
https://github.com/DingXiaoH/RepVGG

The MegEngine version will be released in several days.

TensorRT implemention with C++ API by @upczww https://github.com/upczww/TensorRT-RepVGG. Great work!

Another nice PyTorch implementation by @zjykzj https://github.com/ZJCV/ZCls.

Included in a famous model zoo (over 7k stars) https://github.com/rwightman/pytorch-image-models.

This is a super simple ConvNet architecture that achieves over 80% top-1 accuracy on ImageNet with a stack of 3x3 conv and ReLU! This repo contains the pretrained models, code for building the model, training, and the conversion from training-time model to inference-time.

Citation:

    @article{ding2021repvgg,
    title={RepVGG: Making VGG-style ConvNets Great Again},
    author={Ding, Xiaohan and Zhang, Xiangyu and Ma, Ningning and Han, Jungong and Ding, Guiguang and Sun, Jian},
    journal={arXiv preprint arXiv:2101.03697},
    year={2021}
    }

# Abstract

We present a simple but powerful architecture of convolutional neural network, which has a VGG-like inference-time body composed of nothing but a stack of 3x3 convolution and ReLU, while the training-time model has a multi-branch topology. Such decoupling of the training-time and inference-time architecture is realized by a structural re-parameterization technique so that the model is named RepVGG. On ImageNet, RepVGG reaches over 80\% top-1 accuracy, which is the first time for a plain model, to the best of our knowledge. On NVIDIA 1080Ti GPU, RepVGG models run 83% faster than ResNet-50 or 101% faster than ResNet-101 with higher accuracy and show favorable accuracy-speed trade-off compared to the state-of-the-art models like EfficientNet and RegNet.

![image](https://github.com/DingXiaoH/RepVGG/blob/main/arch.PNG)
![image](https://github.com/DingXiaoH/RepVGG/blob/main/speed_acc.PNG)
![image](https://github.com/DingXiaoH/RepVGG/blob/main/table.PNG)
