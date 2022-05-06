import os
import numpy
import numpy as np
import torch
from torchvision.io import read_image

path = 'assets/MPIIFaceGaze/Image/'

flag = True
total = torch.empty(1)
for i in range(9):
    path_now = path + 'p0' +str(i) + '/left/'
    dir = os.listdir(path_now)
    for img in dir:
        img_path = path_now+img
        eye = read_image(img_path)/255
        if flag:
            total = eye.reshape(-1)
            flag = False
        else:
            total = torch.cat([total, eye.reshape(-1)])

    path_now = path + 'p0' +str(i) + '/right/'
    dir = os.listdir(path_now)
    for img in dir:
        img_path = path_now+img
        eye = read_image(img_path)/255
        total = torch.cat([total, eye.reshape(60*36)])
    print(total.shape)
    print(torch.mean(total))
    print(torch.std(total))
print(torch.mean(total))
print(torch.std(total))
