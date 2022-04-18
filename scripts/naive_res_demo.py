import sys
from os.path import dirname, abspath
path = dirname(dirname(abspath(__file__)))
sys.path.append(path)
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from gaze_estimation.model.resnet import resnet18
from gaze_estimation.utils.Visualization import gaze_visual
from torchvision import transforms
from gaze_estimation.utils.dataloader import load_data
from gaze_estimation.utils.make_loss import angular_error

def get_hlr(data, device):
    """
    Get Hand, Left eyes and Right eyes.
    A function used in trainning part to process data.
    """
    images, labels = data
    faces, lefts, rights = images['Face'], images['Left'], images['Right']
    return faces.to(device), lefts.to(device), rights.to(device), labels.to(device)

# model = torch.load(path + '/assets/model_scripted.pt')
# model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = resnet18(num_classes = 2).to(device).to(device)
model.load_state_dict(torch.load(path + '/assets/model_saved/0.001.pt', map_location=device))
model.eval()

img = Image.open(path + '/assets/test/mine_mid.jpg')
transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
_, val_loader = load_data(128)
for data in val_loader:
    faces, lefts, rights, labels = get_hlr(data, device)
    result = model(faces)
    print(torch.cat([result, labels], dim=1))
    print(angular_error(result, labels))

# plt.figimage(np.array(img))
img = transform(img)
img = torch.unsqueeze(img, 0)
result = model(img)[0].detach().numpy()
result[0] = -result[0]
print(result)
gaze_visual(result)
