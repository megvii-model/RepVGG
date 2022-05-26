# RepVGG: Making VGG-style ConvNets Great Again (CVPR-2021)

The official **PyTorch** implementation, pretrained models and examples are available at 

https://github.com/DingXiaoH/RepVGG

MegEngine version has been included in the MegEngine Basecls model zoo: https://github.com/megvii-research/basecls/tree/main/zoo/public/repvgg

Update (Apr 25, 2021): a deeper RepVGG model achieves **83.55%** top-1 accuracy on ImageNet with SE blocks and an input resolution of 320x320. Note that it is trained with 224x224 but tested with 320x320, so that it is still trainable with a global batch size of 256 on a single machine with 8 1080Ti GPUs. If you test it with 224x224, the top-1 accuracy will be 81.82%. It has 1, 8, 14, 24, 1 layers in the 5 stages respectively. The width multipliers are a=2.5 and b=5 (the same as RepVGG-B2). The model name is "RepVGG-D2se". The PyTorch code for building the model  and testing with 320x320 has been updated and the weights have been released at Google Drive and Baidu Cloud. Please check the PyTorch repo.

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
