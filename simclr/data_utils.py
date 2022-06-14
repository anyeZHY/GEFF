import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torch.utils.data
from gaze.utils.dataloader import get_img, split_mpii, split_columbia, SimData

def make_transform(jitter=0.6, gray=0.2, blur=0.2, sharp=0.2, posterize=0.2):
    color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4, 0.2)
    gaussian_blur = transforms.GaussianBlur(kernel_size=(3,5))
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        transforms.RandomPosterize(2, p=posterize),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        transforms.RandomApply([color_jitter, ], jitter),
        transforms.RandomApply([gaussian_blur, ], blur),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=sharp),
        transforms.RandomGrayscale(p=gray),
    ])
    return transform

def load_data_sim(args, BATCH_SIZE=1024):
    img_dir = ['assets/MPIIFaceGaze/Image', 'assets/ColumbiaGazeCutSet/']
    train, _ = split_mpii(id=7) if args.dataset != 'columbia' else split_columbia(id=42)
    if args.dataset == 'both':
        train = pd.concat([train, split_columbia(id=42)[0]])

    transform_train = make_transform(args.jitter, args.gray)

    color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4, 0.2)

    transform_eye = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((36, 60)),
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5071, std=0.2889),
        transforms.RandomApply([color_jitter, ], args.jitter/2),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=args.sharp),
    ])

    train_set = SimData(train, img_dir, transform_train, transform_eye)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    
    return train_loader
