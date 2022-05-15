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
        self.p_f = nn.Linear(512,512+256)
        self.p_l = nn.Linear(128,128)
        self.p_r = nn.Linear(128,128)
    def forward(self,images):
        face, left, right = images['Face'], images['Left'], images['Right']
        F_f = self.face_en(face)
        F_l = self.eye_en(left)
        F_r = self.eye_en(right)
        return self.p_f(F_f), self.p_l(F_l), self.p_r(F_r)
