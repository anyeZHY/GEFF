from torchvision import transforms
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torch.utils.data
from ge.utils.dataloader import get_img, convert_str_to_float


def make_transform(jitter=0.6, gray=0.2):
    color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        transforms.RandomApply([color_jitter, ], jitter),
        transforms.RandomGrayscale(p=gray),
    ])
    return transform

class SimData(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, transform_eye=None, flip=0.5):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.transform_eye = transform_eye
        self.flip = flip

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_face = get_img(self.img_dir, self.img_labels['Face'].iloc[idx])
        img_left= get_img(self.img_dir, self.img_labels['Left'].iloc[idx])
        img_right = get_img(self.img_dir, self.img_labels['Right'].iloc[idx])

        img_face_i = self.transform(img_face)
        img_left_i = self.transform_eye(img_left)
        img_right_i = self.transform_eye(img_right)

        img_face_j = self.transform(img_face)
        img_left_j = self.transform_eye(img_left)
        img_right_j = self.transform_eye(img_right)

        if torch.rand(1)<self.flip:
            flip = transforms.RandomHorizontalFlip(1)
            img_face_i = flip(img_face)
            img_right_i, img_left_i = flip(img_left), flip(img_right)
            img_face_j = flip(img_face)
            img_right_j, img_left_j = flip(img_left), flip(img_right)

        images_i = {'Face': img_face_i.float(),'Left': img_left_i.float(),'Right': img_right_i.float()}
        images_j = {'Face': img_face_j.float(),'Left': img_left_j.float(),'Right': img_right_j.float()}

        return images_i, images_j

def load_data_sim(BATCH_SIZE=1024):
    train_file = 'assets/MPII_train.csv'
    img_dir = 'assets/MPIIFaceGaze/Image'

    transform_train = make_transform()

    transform_eye = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((60, 36)),
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5071, std=0.2889),
    ])

    train_set = SimData(train_file, img_dir, transform_train, transform_eye)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    
    return train_loader
