import sys
from os.path import dirname, abspath
path = dirname(dirname(abspath(__file__)))
sys.path.append(path)
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from gaze_estimation.model.resnet import ResNet, ResBlock
from gaze_estimation.utils.Visualization import gaze_visual
from torchvision import transforms



# model = torch.load(path + '/assets/model_scripted.pt')
# model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet(ResBlock).to(device)
model.load_state_dict(torch.load(path + '/assets/model_scripted.pt', map_location=device))
model.eval()

img = Image.open(path + '/assets/test/mine.jpg')
transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

# plt.figimage(np.array(img))
img = transform(img)
img = torch.unsqueeze(img, 0)
result = model(img)[0].detach().numpy()
print(result)
gaze_visual(result)
