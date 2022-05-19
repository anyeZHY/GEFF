import cv2
import sys
from os.path import dirname, abspath
path = dirname(dirname(abspath(__file__)))
sys.path.append(path)
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from gaze.utils.Visualization import gaze_visual
from torchvision import transforms
import face_recognition
import time


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(path + '/assets/model_saved/MPII/BaseLr.pt', map_location=torch.device(device))
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

cap = cv2.VideoCapture(0)
# loop runs if capturing has been initialized.
decay = 0.2
weight = 0
smooth = 0
while 1:
    # reads frames from a camera
    ret, img = cap.read()
    W = img.shape[0]
    L = img.shape[1]
    a = 300
    img = img[int(W / 2) - a:int(W / 2) + a, int(L / 2) - a:int(L / 2) + a, :]
    img = img[:,:,[2,1,0]]
    loc = face_recognition.face_locations(img)
    print(loc)
    if len(loc)==0:
        cv2.imshow('img', img)
        continue
    top, right, bottom, left = loc[0]
    img_fed = img[top:bottom, left:right]
    img_tf = transform(img_fed).reshape(1,3,224,224)
    img_tf = {'Face': img_tf}
    result = model(img_tf, None)
    print(result)
    result = result[0].detach().numpy()
    result[0] = - result[0]
    plt.figure()
    plt.imshow(img_fed)

    plt.savefig('figs/face.pdf')
    weight = weight * decay + 1
    smooth = (smooth * decay + result )/ weight
    gaze_visual(smooth*5, show=False)
    plt.title('{}'.format(result))
    plt.savefig('figs/3dgaze.pdf')
    cv2.imshow('img', img[:,:,[2,1,0]])

    # Wait for Esc key to stop
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()

cv2.destroyAllWindows()
