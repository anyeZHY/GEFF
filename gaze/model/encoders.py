# Reference:
# https://github.com/YadiraF/PIXIE/blob/master/pixielib/models/encoders.py

import torch
import torch.nn as nn
from torch import Tensor
from gaze.model.resnet import resnet18

class FaceEncoder(nn.Module):
    def __init__(self, args, idx=9):
        super(FaceEncoder, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.pretrain:
            path = 'assets/model_saved/{}baselineaug.pt'.format(idx) if args.dataset == 'mpii' \
                else 'assets/model_saved/baselinecolumbaseline.pt'
            res = torch.load(path, map_location=torch.device(device)).res
        else:
            res = resnet18(pretrained=True)
        res.fc = nn.Flatten()
        self.res = res
    def forward(self, x):
        return self.res(x)

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

class EyeMLPEncoder(nn.Module):
    def __init__(self, dim_features, in_channels=1):
        super().__init__()
        channel_conv = 1
        self.backbone = MLP((60*36*channel_conv, 1024, dim_features))
    def forward(self, x: Tensor) -> Tensor:
        out = x
        out = out.view(len(out), -1)
        out = self.backbone(out)
        return out

class EyeConvEncoder(nn.Module):
    def __init__(self):
        super(EyeConvEncoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 128, 3 ,padding=1),
            nn.BatchNorm2d(128),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten()
        )

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv(x)
        return out

class EyeResEncoder(nn.Module):
    def __init__(self, eyes_dim=128):
        super().__init__()
        self.f = []
        for name, module in resnet18(num_classes=eyes_dim, channel = [eyes_dim//8, eyes_dim//4, eyes_dim//2, eyes_dim]).named_children():
            if name=='conv1':
                module = nn.Conv2d(1, 64, kernel_size=(7, 7),
                                   stride=(1, 1), padding=(3, 3), bias=False)
            if name=='maxpool':
                continue
            if name=='fc':
                module = nn.Flatten()
            self.f.append(module)
        self.f = nn.Sequential(*self.f)
    def forward(self, x):
        out = self.f(x)
        return out

