# To Do:
# 1. Provide a method to compute the angular loss on ungiven datas
# 2. Compute angluar loss on another dataset (the domain of labels is quite differnent)
import sys
from os.path import dirname, abspath
path = dirname(dirname(abspath(__file__)))
sys.path.append(path)
import argparse
import torch
from torchvision import transforms
from gaze.utils.dataloader import Gaze, make_transform, split_columbia, split_mpii
from torch.utils.data import DataLoader
from gaze.utils.make_loss import angular_error

def make_parser():
    parser = argparse.ArgumentParser(description='Training Congfiguration')
    parser.add_argument("--adapt", action="store_true")
    parser.add_argument("--to", default="columbia", choices=['mpii', 'columbia'], type=str)
    return parser


def validate(val_loader, path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(path, map_location=torch.device(device))
    model.eval()

    aug_loss_total = 0
    total = 0
    L = len(val_loader)
    with torch.no_grad():
        for i,data in enumerate(val_loader,1):
            imgs, labels = data
            labels = labels.to(device)
            imgs = {name: imgs[name].to(device) for name in imgs}
            gaze = model(imgs)
            total += labels.size(0)
            loss = angular_error(gaze, labels.float())
            aug_loss_total += loss.item() * (labels.size(0))
            print('\rTesting... %.03f' % (i/L*100) + '%', end='', flush=True)
    print('\nTest\'s AngularLoss is: %.03f' % (aug_loss_total / total))


def domain_adaptation(args):
    img_dir = 'assets/'
    if args.to=='columbia':
        val, _ = split_columbia(100)
        img_dir_val = img_dir + 'ColumbiaGazeCutSet'
        path = 'assets/model_saved/MPII/geffrf_heavy.pt'
    else:
        val, _ = split_mpii(100)
        img_dir_val = img_dir + 'MPIIFaceGaze/Image'
        path = 'assets/model_saved/Columbia/geffmfp_warm.pt'
    transform_eye, transform_val = make_transform()
    val_set = Gaze(val, img_dir_val, transform_val, transform_eye, flip=0)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=128, shuffle=True)
    validate(val_loader, path)



def test():
    img_dir = 'assets/'
    img_dir_val = img_dir + 'MPIIFaceGaze/Image'
    val, _ = split_mpii(id=100, start=10, end=15)
    transform_eye, transform_val = make_transform()
    val_set = Gaze(val, img_dir_val, transform_val, transform_eye, flip=0)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=128, shuffle=True)
    path = [
        'assets/model_saved/MPII/geffrf_heavay.pt',
        'assets/model_saved/MPII/geffmf100.pt',
        'assets/model_saved/MPII/fusemfull.pt',
        'assets/model_saved/MPII/Base.pt',
        'assets/model_saved/MPII/geffsim.pt',
    ]
    for model_path in path:
        print(model_path)
        validate(val_loader, model_path)
    print("MPII dataset test finished successfully.")


def main():
    args = make_parser().parse_args()
    if args.adapt:
        domain_adaptation(args)
    else:
        test()
    return

if __name__ == '__main__':
    main()
