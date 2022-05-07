import torch
from torchvision import transforms
from torchvision.io import read_image
import numpy as np
from time import sleep
import matplotlib.pyplot as plt
import matplotlib.animation as ani

def make_transform(flip=0.6, jitter=0.6, gray=0.2):
    color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        # transforms.RandomHorizontalFlip(flip),
        # transforms.RandomVerticalFlip(flip),
        transforms.RandomApply([color_jitter, ], jitter),
        # transforms.RandomRotation(degrees=(0, 180)),
        transforms.RandomGrayscale(p=gray),
    ])
    return transform
