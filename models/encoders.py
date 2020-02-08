import torch
import torch.nn as nn

from models.backbones.mobilenet import MobileNet
from models.backbones.vanila_mobilenet import VanilaMobileNet

from models.backbones.nasnet_mobile import NASNetAMobile

from models.backbones.resnet import ResNet
from models.backbones.resnet import BasicBlock
from models.backbones.resnet import Bottleneck

from models.backbones.senet import SENet
from models.backbones.senet import SEBottleneck
from models.backbones.senet import SEResNetBottleneck
from models.backbones.senet import SEResNeXtBottleneck

from models.backbones.xception import Xception


def mobilenet(device='cpu', *argv, **kwargs):
    model = MobileNet(*argv, **kwargs)
    return model.to(device)


def vanila_mobilenet(device='cpu', *argv, **kwargs):
    model = VanilaMobileNet(*argv, **kwargs)
    return model.to(device)


def nasnet_mobile(device='cpu', *argv, **kwargs):
    model = NASNetAMobile(*argv, **kwargs)
    return model.to(device)


def resnet18(device='cpu', *argv, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model.to(device)


def resnet34(device='cpu', *argv, **kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model.to(device)


def resnet50(device='cpu', *argv, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model.to(device)


def resnet101(device='cpu', *argv, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model.to(device)


def resnet152(device='cpu', *argv, **kwargs):
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model.to(device)


def senet154(device='cpu', *argv, **kwargs):
    model = SENet(SEBottleneck, [3, 8, 36, 3], groups=64, reduction=16,
                  **kwargs)
    return model.to(device)


def se_resnet50(device='cpu', *argv, **kwargs):
    model = SENet(SEResNetBottleneck, [3, 4, 6, 3], groups=1, reduction=16,
                  inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0,
                  **kwargs)
    return model.to(device)


def se_resnet101(device='cpu', *argv, **kwargs):
    model = SENet(SEResNetBottleneck, [3, 4, 23, 3], groups=1, reduction=16,
                  inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0,
                  **kwargs)
    return model.to(device)


def se_resnet152(device='cpu', *argv, **kwargs):
    model = SENet(SEResNetBottleneck, [3, 8, 36, 3], groups=1, reduction=16,
                  inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0,
                  **kwargs)
    return model.to(device)


def se_resnext50(device='cpu', *argv, **kwargs):
    model = SENet(SEResNeXtBottleneck, [3, 4, 6, 3], groups=32, reduction=16,
                  inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0,
                  **kwargs)
    return model.to(device)


def se_resnext101(device='cpu', *argv, **kwargs):
    model = SENet(SEResNeXtBottleneck, [3, 4, 23, 3], groups=32, reduction=16,
                  inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0,
                  **kwargs)
    return model.to(device)


def xception(device='cpu', *argv, **kwargs):
    model = Xception(*argv, **kwargs)
    return model.to(device)
