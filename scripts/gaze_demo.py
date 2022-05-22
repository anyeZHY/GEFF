import os
import sys
from os.path import dirname, abspath
path = dirname(dirname(abspath(__file__)))
sys.path.append(path)
import torch
import matplotlib.pyplot as plt
import numpy as np
from gaze.utils.dataloader import load_data
from gaze.utils.make_loss import angular_error
import imageio


def gaze_show(gaze):
    # circle
    theta = np.linspace(-np.pi,np.pi)
    plt.plot(np.cos(theta), np.sin(theta))
    plt.plot([0, 0], [-1, 1], c='#1f77b4')
    plt.plot([-1, 1], [0, 0], c='#1f77b4')
    # Gaze
    yaw, pitch = gaze
    plt.arrow(0, 0, np.sin(yaw) * np.cos(pitch), np.sin(pitch), width=0.01)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(path + '/assets/model_saved/MPII/geffmf.pt', map_location=torch.device(device))
model.eval()

_, val_loader = load_data(None, 2, val_size=1, flip=1)
filenames = {}
for i, data in enumerate(val_loader,1):
    imgs, labels = data
    result = model(imgs, None)

    # print(torch.cat([result, labels], dim=1))
    augloss = angular_error(result, labels, every=True)[0].detach().numpy()
    labels = labels.detach().numpy()
    result = result.detach().numpy()

    face = imgs['Face'].reshape(3,224,224).permute(1,2,0).detach().numpy()
    face *= [0.229, 0.224, 0.225]
    face += [0.485, 0.456, 0.406]
    face = np.clip(face, a_min=0, a_max=1)
    plt.figure(figsize=(12,5))
    # Gaze
    plt.subplot(1, 2, 1)
    gaze = result[0]
    gaze[0] = -gaze[0]
    gaze_show(2 * gaze)
    plt.title('Predict: %.03f, %.03f' % (-result[0][0], result[0][1]))
    plt.text(-0.428, 0.4, 'Auglar loss: %.03f degree' % augloss )
    # Face
    plt.subplot(1, 2, 2)
    plt.imshow(face, vmin=0)
    plt.title('Ground truth: %.03f, %.03f'% (labels[0][0], labels[0][1]))
    filename = 'figs/gaze{}.png'.format(i)
    filenames[filename] = float(augloss)
    plt.savefig(filename)
    print('\r' + '#' * (i+1), end="")
    if i == 20:
        break

# build gif
images = []
filenames = sorted(filenames.items(), key=lambda x : x[1])
for filename in filenames[0:10]:
    image = imageio.imread(filename[0])
    images.append(image)
imageio.mimsave('figs/gaze_demo.gif', images, duration = 1.1)
for filename in set(filenames):
    os.remove(filename[0])
