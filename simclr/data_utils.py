import torch
from torchvision import transforms
from torchvision.io import read_image
import numpy as np
from time import sleep
from celluloid import Camera
import matplotlib.pyplot as plt
import matplotlib.animation as ani

def make_transform(flip=0.6, jitter=0.6, gray=0.2):
    color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(flip),
        transforms.RandomVerticalFlip(flip),
        transforms.RandomApply([color_jitter, ], jitter),
        transforms.RandomRotation(degrees=(0, 180)),
        transforms.RandomGrayscale(p=gray),

    ])
    return transform

if __name__ == '__main__':
    fig = plt.figure()
    ax = plt.axes()
    ax.axis('off')
    camera = Camera(fig)
    for i in range(100):
        img = read_image('simclr/test.jpg')
        transform_test = make_transform()
        img_transformed = transform_test(img).permute(1, 2, 0)
        ax.imshow(img_transformed)
        plt.pause(0.1)
        camera.snap()

    animation = camera.animate()
    animation.save('simclr/test.gif', writer='PillowWriter', fps=3)
