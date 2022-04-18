import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torchvision.io import read_image, image
from torch.utils.data import Dataset
from torchvision import transforms
# from sklearn.utils import shuffle
import torch.utils.data

datapath = 'assets/MPIIFaceGaze/'
colomn = ['Face', 'Left', 'Right', '3DGaze', '2DGaze', '3DHead', '2DHead']

# ============== Useful functions >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def convert_str_to_float(string):
    """
    convert the string to a darray
    """
    return np.array(list(map(float, string.split(','))))
def get_img(dir, path, mode='rgb'):
    path = os.path.join(dir, path)
    if mode=='rgb':
        return read_image(path)
    else:
        return read_image(path, mode=image.ImageReadMode.GRAY)

# ============== Process .label >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def split_data(test_length = 1000, val_length = 5000):
    """
    Split/resplit the data into training set, validation set and test set.
    Nota Bene: we will shuffle the dataset first.
    """
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
            # for col in ['3DGaze', '2DGaze', '3dHead', '2DHead']:
            #     arr = np.array(list(map(float, df[col][i_pic].split(','))))
            #     df[col][i_pic] = arr
        data = pd.concat([data, df])
    data = data[colomn].sample(frac = 1).reset_index(drop=True)
    # print(data.head())
    data.iloc[0:test_length].to_csv('assets/MPII_test.csv')
    data.iloc[test_length+1:test_length+val_length].to_csv('assets/MPII_val.csv')
    data.tail(len(data)-test_length-val_length).reset_index(drop=True).to_csv('assets/MPII_train.csv')
    return data

# ============== Data Process >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
class MPII(Dataset):
    """
    Use torch.utils.data.Dataset to process MPII dataset.
    Naive version: label is '2DGaze'.
    """
    def __init__(self, annotations_file, img_dir,
                 transform=None, transform_eye=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.transform_eye = transform_eye

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_face = get_img(self.img_dir, self.img_labels['Face'].iloc[idx])
        img_left= get_img(self.img_dir, self.img_labels['Left'].iloc[idx])/255
        img_right = get_img(self.img_dir, self.img_labels['Right'].iloc[idx])/255
        label = convert_str_to_float(self.img_labels['2DGaze'].iloc[idx])
        if self.transform:
            img_face = self.transform(img_face)
        if self.transform_eye:
            img_left = self.transform_eye(img_left)
            img_right = self.transform_eye(img_right)
        if self.target_transform:
            label = self.target_transform(label)
        images = {'Face': img_face.float(), 'Left': img_left.float(), 'Right': img_right.float()}
        return images, label.astype(float)

def load_data(BATCH_SIZE, transform_train=None):
    # df_data = procees_data(0)
    # df_data = pd.read_pickle('assets/MPII_2D_annoataion.csv')
    if not os.path.isfile('assets/MPII_test.csv'):
        split_data()
    train_file = 'assets/MPII_train.csv'
    val_file = 'assets/MPII_val.csv'
    img_dir = 'assets/MPIIFaceGaze/Image'

    if transform_train is None:
        transform_train = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            # transforms.RandomApply([
            #     transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            # ], p=0.8)
        ])
    transform_eye = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((60, 36)),
        transforms.ToTensor(),
        transforms.Normalize(mean=0.505, std=0.084),
    ])

    transform_val = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_set = MPII(train_file, img_dir, transform_train, transform_eye)
    val_set = MPII(val_file, img_dir, transform_val, transform_eye)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=100, shuffle=True)

    return train_loader, val_loader

if __name__ == '__main__':
    # load_data_naive(128)
    split_data()
