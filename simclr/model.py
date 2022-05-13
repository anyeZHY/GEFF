import torch
import torch.nn as nn
from ge.model.resnet import resnet18
from ge.model.encoders import EyeResEncoder

class SimCLR(nn.Module):
    def __init__(self):
        super(SimCLR, self).__init__()
        res = resnet18()
        res.fc = nn.Flatten()
        self.face_en = res
        self.eye_en = EyeResEncoder()
SimCLR()
