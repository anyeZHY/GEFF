import os
import pandas as pd
import numpy as np
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

def split_mpii(id: int):
    """
    Split/resplit the data into training set, validation set and test set.
    """
    print("Loading Data...")
    train = pd.DataFrame(columns=colomn)
    val = pd.DataFrame(columns=colomn)
    for i_label in range(10):
        labelpath = datapath + 'Label/p' + str(i_label).zfill(2) + '.label'
        df = pd.read_table(labelpath, delimiter=' ')
        df = df[colomn]
        for i_pic in range(len(df)):
            for col in ['Face', 'Left', 'Right']:
                facepath = df[col][i_pic].replace('\\','/')
                df[col][i_pic] = facepath
        if i_label == id:
            val = pd.concat([val, df])
        else:
            train = pd.concat([train, df])
    train= train[colomn].sample(frac = 1).reset_index(drop=True)
    val= val[colomn].sample(frac = 1).reset_index(drop=True)
    # train.to_csv('assets/MPII_train.csv')
    # val.to_csv('assets/MPII_val.csv')
    print("Done!")
    return train, val

# ============== Data Process >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

def mask(face, eye):
    shape = eye.shape
    p = torch.rand(1)
    if 0 < p < 0.05:
        eye = torch.randn(shape)
    elif 0.05 <= p < 0.2 :
        eye = transforms.RandomCrop(shape[1:])(face)
        eye = eye[0].reshape(1, 36 ,60)
    else:
        color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        gaussian_blur = transforms.GaussianBlur(kernel_size=(3, 5))
        T = transforms.Compose([
            transforms.RandomApply([color_jitter, ], 0.2),
            transforms.RandomApply([gaussian_blur, ], 0.2),
            transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.2),
        ])
        eye = T(eye)
    return eye

class MPII(Dataset):
    """
    Use torch.utils.data.Dataset to process MPII dataset.
    Version: label is '2DGaze'.
    """
    def __init__(self, annotations_file, img_dir,
                 transform=None, transform_eye=None, target_transform=None, flip=0, use_mask=False):
        self.img_labels = annotations_file
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.transform_eye = transform_eye
        self.flip = flip
        self.mask = use_mask

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_face = get_img(self.img_dir, self.img_labels['Face'].iloc[idx])
        img_left= get_img(self.img_dir, self.img_labels['Left'].iloc[idx])
        img_right = get_img(self.img_dir, self.img_labels['Right'].iloc[idx])
        label = convert_str_to_float(self.img_labels['2DGaze'].iloc[idx])
        if self.transform:
            img_face = self.transform(img_face)
        if self.transform_eye:
            img_left = self.transform_eye(img_left)
            img_right = self.transform_eye(img_right)
        if self.target_transform:
            label = self.target_transform(label)
        if torch.rand(1)<self.flip:
            flip = transforms.RandomHorizontalFlip(1)
            label[:][0] = - label[:][0]
            img_face = flip(img_face)
            img_right, img_left = flip(img_left), flip(img_right)
        if self.mask:
            img_left = mask(img_face, img_left)
            img_right = mask(img_face, img_right)
        images = {
            'Face': img_face.float(),
            'Left': img_left.float(),
            'Right': img_right.float()
        }
        return images, label.astype(float)

def make_transform():
    transform_eye = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((36, 60)),
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5071, std=0.2889),
    ])

    transform_val = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    return transform_eye, transform_val


def load_data(args, BATCH_SIZE, val_size=128, transform_train=None, flip=0, person_id=9):
    use_mask = False if args is None else args.mask
    train_file, val_file, img_dir = 'assets/', 'assets/', 'assets/'
    train, val = None, None
    if args is None or args.dataset == 'mpii':
        train, val = split_mpii(person_id)
        img_dir += 'MPIIFaceGaze/Image'
    else:
        pass
    if transform_train is None:
        transform_train = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    transform_eye, transform_val = make_transform()

    train_set = MPII(train, img_dir, transform_train, transform_eye, flip=flip, use_mask=use_mask)
    val_set = MPII(val, img_dir, transform_val, transform_eye, flip=0)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=val_size, shuffle=True)
    return train_loader, val_loader

if __name__ == '__main__':
    split_mpii(9)
