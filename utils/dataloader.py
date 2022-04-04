import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision import transforms
import torch.utils.data

datapath = 'assets/MPIIFaceGaze/'
colomn = ['Face', 'Left', 'Right', '3DGaze', '2DGaze']

# unused now
def procees_data(save = 1):
    data = pd.DataFrame(columns=colomn)
    for i_label in range(10):
        labelpath = datapath + 'Label/p' + str(i_label).zfill(2) + '.label'
        df = pd.read_table(labelpath, delimiter=' ')
        # df = df.head()
        df = df[colomn]
        # print(len(df))
        for i_pic in range(len(df)):
            for col in ['Face', 'Left', 'Right']:
                facepath = datapath + 'Image/' + df[col][i_pic].replace('\\','/')
                im = np.array(Image.open(facepath))
                df[col][i_pic] = im
            for col in ['3DGaze', '2DGaze']:
                arr = np.array(list(map(float, df[col][i_pic].split(','))))
                df[col][i_pic] = arr
        data = pd.concat([data, df])
    if save:
        data.to_pickle('assets/MPIIFaceGazeData.csv')
    return data

def extract_from_data(save = 1):
    data = pd.DataFrame(columns=colomn)
    for i_label in range(10):
        labelpath = datapath + 'Label/p' + str(i_label).zfill(2) + '.label'
        df = pd.read_table(labelpath, delimiter=' ')
        # df = df.head()
        df = df[colomn]
        # print(len(df))
        for i_pic in range(len(df)):
            for col in ['Face', 'Left', 'Right']:
                facepath = df[col][i_pic].replace('\\','/')
                # im = np.array(Image.open(facepath))
                df[col][i_pic] = facepath
            for col in ['3DGaze', '2DGaze']:
                arr = np.array(list(map(float, df[col][i_pic].split(','))))
                df[col][i_pic] = arr
        data = pd.concat([data, df])
    if save:
        data[['Face', '3DGaze']].to_pickle('assets/MPII_3D_annoataion.csv')
        data[['Face', '2DGaze']].to_pickle('assets/MPII_2D_annoataion.csv')
    return data

# ============== Data Process >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Reference: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files
class MPII(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_pickle(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image.float(), label.astype(float)

def load_data_naive(BATCH_SIZE):
    # df_data = procees_data(0)
    # df_data = pd.read_pickle('assets/MPII_2D_annoataion.csv')
    annotations_file = 'assets/MPII_2D_annoataion.csv'
    img_dir = 'assets/MPIIFaceGaze/Image'

    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        # transforms.RandomApply([
        #     transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        # ], p=0.8)
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    data = MPII(annotations_file, img_dir, transform_train)

    train_set, val_set = torch.utils.data.random_split(data, [25000, 5000])

    trainloader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)

    testloader = torch.utils.data.DataLoader(val_set, batch_size=100, shuffle=True)
    return trainloader, testloader

if __name__ == '__main__':
    # load_data_naive(128)
    # procees_data()
    extract_from_data()
