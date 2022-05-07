from torchvision import transforms

def make_transform(flip=0.6, jitter=0.6, gray=0.2):
    color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        # transforms.RandomHorizontalFlip(flip),
        # transforms.RandomVerticalFlip(flip),
        transforms.RandomApply([color_jitter, ], jitter),
        # transforms.RandomRotation(degrees=(0, 180)),
        transforms.RandomGrayscale(p=gray),
    ])
    return transform
