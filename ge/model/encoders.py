# Reference:
# https://github.com/YadiraF/PIXIE/blob/master/pixielib/models/encoders.py

# import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import torchvision.models as models

class ResnetEncoder(nn.Module):
    def __init__(self, append_layers=None):
        super(ResnetEncoder, self).__init__()
        import resnet
        # feature_size = 2048
        self.feature_dim = 2048
        self.encoder = resnet.resnet18()  # out: 2048
        # regressor
        self.append_layers = append_layers
        # for normalize input images
        MEAN = [0.485, 0.456, 0.406]
        STD = [0.229, 0.224, 0.225]
        self.register_buffer('MEAN', torch.tensor(MEAN)[None, :, None, None])
        self.register_buffer('STD', torch.tensor(STD)[None, :, None, None])

    def forward(self, inputs):
        """
        inputs: [bz, 3, h, w], range: [0,1]
        """
        inputs = (inputs - self.MEAN) / self.STD
        features = self.encoder(inputs)
        if self.append_layers:
            features = self.last_op(features)
        return features


class MLP(nn.Module):
    def __init__(self, channels=(2048, 1024, 1), last_op=None):
        super(MLP, self).__init__()
        layers = []

        for l in range(0, len(channels) - 1):
            layers.append(nn.Linear(channels[l], channels[l + 1]))
            if l < len(channels) - 2:
                layers.append(nn.BatchNorm1d(channels[l+1]))
                layers.append(nn.ReLU())
        if last_op:
            layers.append(last_op)

        self.layers = nn.Sequential(*layers)

    def forward(self, inputs):
        outs = self.layers(inputs)
        return outs

class EyeResEncoder(nn.Module):
    def __init__(self, dim_features, in_channels=1):
        from .resnet import resnet18
        super().__init__()
        channel_conv = 1
        # self.conv = nn.Conv2d(in_channels, channel_conv, 3, padding=1)
        # self.bn = nn.BatchNorm2d(channel_conv)
        # self.relu = nn.ReLU(inplace=True)
        self.backbone = MLP((60*36*channel_conv, 1024, dim_features))
        # self.res = resnet18(num_classes=dim_features)
    def forward(self, x: Tensor) -> Tensor:
        out = x
        # out = self.conv(out)
        # out = self.bn(out)
        # out = self.relu(out)
        out = out.view(len(out), -1)
        out = self.backbone(out)
        return out

