# import pandas as pd
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
# import torch.nn.functional as F
# import torchvision
from gaze_estimation.utils.dataloader import load_data
from gaze_estimation.utils.make_loss import angular_error
from gaze_estimation.model.resnet import resnet18
from gaze_estimation.model.encoders import MLP, EyeResEncoder

def get_hlr(data, device):
    """
    Get Hand, Left eyes and Right eyes.
    A function used in trainning part to process data.
    """
    images, labels = data
    faces, lefts, rights = images['Face'], images['Left'], images['Right']
    return faces.to(device), lefts.to(device), rights.to(device), labels.to(device)

def train(args):
    """
    Input
    - arg: Arguments of:
        - epoch: Default = 10
        - lr: Learning rate. Default = 1e-3
        - out_channel: The size of output. Default = 2
        - res_channels: The channels of ResNet, of shape (4, ). Default = (16, 32, 64, 128)
    """
    # check gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # set hyperparameter
    print_every = args.print_every if (not args.debug) else 1
    EPOCH = args.epoch if (not args.debug) else 1
    BATCH_SIZE = args.batch if (not args.debug) else 16
    LR = args.lr
    out_channel = args.out_channel

    # prepare dataset and preprocessing
    train_loader, val_loader = load_data(BATCH_SIZE)

    # define ResNet18
    dim_face = 512
    dim_eyes = 128
    encoder_face = resnet18(num_classes=dim_face).to(device)
    encoder_eye = EyeResEncoder(dim_features=dim_eyes).to(device)
    MLP_channels = (dim_face + dim_eyes * 2, out_channel)
    decoder = MLP(channels=MLP_channels).to(device)

    L1 = nn.SmoothL1Loss(reduction='mean')
    optimizer = optim.Adam(encoder_face.parameters(), lr=LR)

    for epoch in range(0, EPOCH):
        print('\nEpoch: %d' % (epoch + 1))
        encoder_face.train()
        encoder_eye.train()
        decoder.train()
        sum_loss = 0.0
        length = len(train_loader)
        for i, data in enumerate(train_loader, 0):
            # prepare dataset
            faces, lefts, rights, labels =  get_hlr(data, device)
            optimizer.zero_grad()
            # forward & backward
            F_face = encoder_face(faces) # Feature_face
            F_left = encoder_eye(lefts)
            F_right = encoder_eye(rights)
            features = torch.cat((F_face, F_left, F_right), dim=1)
            gaze = decoder(features)

            ang_loss = angular_error(gaze, labels)
            L1_loss = L1(gaze, labels.float())
            loss = ang_loss + 0.05 * L1_loss
            loss.backward()
            optimizer.step()
            # torch.nn.utils.clip_grad_norm_(encoder_face.parameters(), 5) #

            # print ac & loss in each batch
            sum_loss += ang_loss.item()
            if (i+1+epoch*length)%print_every == 0:
                print('[epoch:%d, iter:%d] Loss: %.03f '
                    % (epoch + 1, (i + 1 + epoch * length), loss.item()))
            if args.debug:
                break

        # get the ac with testdataset in each epoch
        print('Waiting Test...')
        encoder_face.eval()
        encoder_eye.eval()
        decoder.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for data in val_loader:
                faces, lefts, rights, labels = get_hlr(data, device)
                F_face = encoder_face(faces)  # Feature_face
                F_left = encoder_eye(lefts)
                F_right = encoder_eye(rights)
                features = torch.cat((F_face, F_left, F_right), dim=1)
                gaze = decoder(features)
                total += labels.size(0)

                loss = angular_error(gaze, labels.float())
                # loss = criterion(yaw_pitch_to_vec(F_face), yaw_pitch_to_vec(labels.float()))
                correct += loss.item() * (labels.size(0))
                if args.debug:
                    break
            print('Test\'s loss is: %.03f' % (correct/total))

        if args.debug:
            break
        filename = 'assets/model_saved/' + args.name + \
                   ',lr={lr},' \
                   'total_epoch={epoch},' \
                   'epoch_save={now},' \
                   '.pt'.format(
                       lr=LR, epoch=EPOCH, now=epoch+1
                   )
        if args.save_every:
            torch.save(encoder_face.state_dict(), filename)

    print('Train has finished, total epoch is %d' % EPOCH)
    filename = 'assets/model_saved/' + args.name + \
               ',lr={lr},' \
               'total_epoch={epoch},' \
               '.pt'.format(
                   lr=LR, epoch=EPOCH
               )
    if not args.debug:
        torch.save(encoder_face.state_dict(), filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training Congfiguration')

    parser.add_argument("--name", default="Name", type=str, help='Config File names')
    parser.add_argument("--debug", action="store_true", help="Train concisely and roughly")
    parser.add_argument("--save_every", action="store_true", help="Save models after every epoch")
    parser.add_argument("--print_every", default=50, type=int, help="Print loss")

    # hyperparameters in Trainnig part
    parser.add_argument("--epoch", default=10, type=int)
    parser.add_argument("--batch", default=128, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--out_channel", default=2, type=int)
    parser.add_argument("--res_channels", nargs='+', default=[16, 32, 64, 128], type=int)
    train(parser.parse_args())
