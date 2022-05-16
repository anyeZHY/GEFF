import sys
from os.path import dirname, abspath
path = dirname(dirname(abspath(__file__)))
sys.path.append(path)
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from ge.utils.Visualization import gaze_visual
from torchvision import transforms
from ge.utils.dataloader import load_data
from ge.utils.make_loss import angular_error
import time


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(path + '/assets/model_saved/baseline/BaseLr.pt', map_location=torch.device(device))
model.eval()

img = Image.open(path + '/assets/test/mine_left.jpg')
transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
_, val_loader = load_data(128, val_size=1, flip=1)
for data in val_loader:
    imgs, labels = data
    result = model(imgs, None)

    # print(torch.cat([result, labels], dim=1))
    print(angular_error(result, labels, every=True))
    labels = labels.detach().numpy()
    result = result.detach().numpy()
    print(result)
    print(labels)

    face = imgs['Face'].reshape(3,224,224).permute(1,2,0).detach().numpy()
    face *= [0.229, 0.224, 0.225]
    face += [0.485, 0.456, 0.406]
    plt.figure()
    plt.imshow(face)
    plt.title('Gaze: %.03f, %.03f'% (labels[0][0], labels[0][1]))
    plt.savefig('figs/face.pdf')
    gaze = result[0]
    gaze[0] = -gaze[0]
    gaze_visual(2*gaze, show=False)
    plt.title('Result: %.03f, %.03f' % (-result[0][0], result[0][1]))
    plt.savefig('figs/3dgaze.pdf')
    time.sleep(5)

# plt.figimage(np.array(img))
img = transform(img)
img = torch.unsqueeze(img, 0)
img = {'Face': img}
result = model(img)[0].detach().numpy()
result[0] = -result[0]
print(result)

