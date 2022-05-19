import shutil
import os
from torchvision.io import read_image
import torchvision.transforms as T
import torchvision.transforms.functional as F
import cv2


def create_folders():
    path = os.getcwd() + '/assets/ColumbiaGazeCutSet/'
    folder = os.listdir(path)
    for curr_dir in folder:
        if curr_dir == '.DS_Store':
            continue
        src = path + curr_dir + '/'
        dst = [src + 'face/', src + 'left/', src + 'right/']
        for d in dst:
            if not os.path.isdir(d):
                os.mkdir(d)
        for file in os.listdir(src):
            if file == 'face' or file == 'left' or file == 'right':
                continue
            shutil.move(src + file, dst[0] + file)


def cut_eye():
    path = os.getcwd() + '/assets/ColumbiaGazeCutSet/'
    folder = os.listdir(path)
    for curr_dir in folder:
        if curr_dir == '.DS_Store':
            continue
        files = os.listdir(path+curr_dir+'/face')
        for file in files:
            img = read_image(path+curr_dir+'/face/'+file)
            left = F.crop(img, 50, 30, 36, 60)
            right = F.crop(img, 50, 115, 36, 60)
            left_gray = T.Grayscale()(left).detach().numpy()
            right_gray = T.Grayscale()(right).detach().numpy()
            cv2.imwrite(path+curr_dir+'/left/'+file, left_gray.reshape(36, 60, 1))
            cv2.imwrite(path+curr_dir+'/right/'+file, right_gray.reshape(36, 60, 1))


if __name__ == '__main__':
    create_folders()  
    cut_eye()


