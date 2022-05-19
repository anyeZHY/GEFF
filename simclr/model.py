import torch
import torch.nn as nn
import torch.nn.functional as F
from gaze.model.resnet import resnet18
from gaze.model.encoders import EyeResEncoder

class SimCLR(nn.Module):
    def __init__(self):
        super(SimCLR, self).__init__()
        res = resnet18(pretrained=True)
        res.fc = nn.Flatten()
        self.face_en = res
        self.eye_en = EyeResEncoder()
        self.p_f = nn.Sequential(
            nn.Linear(512,512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512+256, bias=True)
        )
        self.p_l = nn.Sequential(
            nn.Linear(128,128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128, bias=True)
        )
        self.p_r = nn.Sequential(
            nn.Linear(128,128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128, bias=True)
        )
    def forward(self,images):
        face, left, right = images['Face'], images['Left'], images['Right']
        F_f = self.face_en(face)
        F_l = self.eye_en(left)
        F_r = self.eye_en(right)
        out_f = self.p_f(F_f)
        out_l = self.p_l(F_l)
        out_r = self.p_r(F_r)
        return F.normalize(out_f, dim=-1), F.normalize(out_l, dim=-1), F.normalize(out_r, dim=-1)
