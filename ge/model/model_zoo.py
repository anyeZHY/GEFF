import sys
from os.path import dirname, abspath
path = dirname(dirname(abspath(__file__)))
sys.path.append(path)

import torch
import torch.nn as nn
from ge.model.resnet import resnet18
from ge.model.geff import GEFF
from ge.model.encoders import MLP, EyeEncoder, EyeResEncoder

class ResGazeNaive(nn.Module):
    def __init__(self):
        super().__init__()
        self.res = resnet18(num_classes=2)

    def forward(self, imgs):
        face = imgs['Face']
        out = self.res(face)
        return out


class Fuse(nn.Module):
    def __init__(self, models, weight=0.2, share_eye=False):
        super().__init__()
        self.face_en = models['Face']
        self.share_eye = share_eye
        if share_eye:
            self.eye = models['Eye']
        else:
            self.left_en = models['Left']
            self.right_en = models['Right']
        self.decoder = models['Decoder']
        self.w = weight

    def forward(self, imgs, similarity=False):
        faces, lefts, rights = imgs['Face'], imgs['Left'], imgs['Right']
        # forward & backward
        F_face = self.face_en(faces)  # Feature_face
        _, D = F_face.shape
        if self.share_eye:
            F_left = self.eye(lefts)
            F_right = self.eye(lefts)
        else:
            F_left = self.left_en(lefts)
            F_right = self.right_en(rights)
        F_lf = F_left * self.w + F_face[:, :D//4] * (1 - self.w)
        F_rf = F_right * self.w + F_face[:, D // 4:D // 2] * (1 - self.w)
        features = torch.cat((F_face[:, D//2:], F_lf, F_rf), dim=1)
        gaze = self.decoder(features)
        if similarity:
            return gaze, F_left, F_right, F_face[:, :D//4], F_face[:, D // 4:D // 2]
        else:
            return gaze


def get_model(args, models=None, share_eye=False):
    name = args.model
    if name == 'baseline':
        return ResGazeNaive()
    if name == 'geff':
        return GEFF(models, share_eye=share_eye)
    if name == 'fuse':
        return Fuse(models, share_eye=share_eye)


def gen_geff(args, channels = None, device=None):
    """
    Args
        channels: dictionary
            - Face: nteger
            - Out: integer
            - Fussion: tuple
    """
    name = args.model
    models = None
    fussion_channel = None
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if channels is None:
        face_dim = 512
        eyes_dim = face_dim // 4
        out_channel = 2
        fussion_channel = (face_dim//2, 128, 1)
        decoder_channel = (face_dim, 128, out_channel)
    else:
        face_dim = channels['Face']
        eyes_dim = face_dim // 4
        out_channel = channels['Out']
        decoder_channel = (face_dim, 128, out_channel)
        if name == 'geff':
            fussion_channel = channels['Fussion']
    if name == 'fuse':
        models = {
            'Face': resnet18(num_classes = face_dim).to(device),
            'Left': EyeEncoder(dim_features=eyes_dim).to(device),
            'Right': EyeEncoder(dim_features=eyes_dim).to(device),
            'Decoder': MLP(channels = decoder_channel).to(device),
            'Eye': EyeResEncoder(eyes_dim).to(device),
        }
    if name == 'geff':
        models = {
            'Face': resnet18(num_classes=face_dim).to(device),
            'Left': EyeEncoder(dim_features=eyes_dim).to(device),
            'Right': EyeEncoder(dim_features=eyes_dim).to(device),
            'Eye': EyeResEncoder(eyes_dim).to(device),
            'Extractor': MLP(face_dim, eyes_dim).to(device),
            'Fussion_l': MLP(channels=fussion_channel).to(device),
            'Fussion_r': MLP(channels=fussion_channel).to(device),
            'Decoder': MLP(channels=decoder_channel).to(device),
        }
    return models
