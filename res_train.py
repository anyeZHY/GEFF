# import pandas as pd
import argparse
import torch
# import torch.nn as nn
import torch.optim as optim
# import torch.nn.functional as F
# import torchvision
from gaze_estimation.utils.dataloader import load_data
from gaze_estimation.utils.make_loss import angular_error
from gaze_estimation.model.resnet import ResNet, ResBlock

def train(args):
    """
    # naive trainnig part
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
    EPOCH = args.epoch
    pre_epoch = 0
    BATCH_SIZE = args.batch
    LR = args.lr
    out_channel = args.out_channel
    res_channels = args.res_channels

    # prepare dataset and preprocessing
    train_loader, val_loader = load_data(BATCH_SIZE)

    # define ResNet18
    model = ResNet(ResBlock, out_channel=out_channel, channels=res_channels).to(device)

    # criterion = nn.SmoothL1Loss(reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(pre_epoch, EPOCH):
        print('\nEpoch: %d' % (epoch + 1))
        model.train()
        sum_loss = 0.0
        length = len(train_loader)
        for i, data in enumerate(train_loader, 0):
            # prepare dataset
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            # forward & backward
            outputs = model(inputs)
            loss = angular_error(outputs, labels.float())
            loss.backward()
            optimizer.step()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5) #

            # print ac & loss in each batch
            sum_loss += loss.item()
            if (i+1+epoch*length)%args.print_every == 0:
                print('[epoch:%d, iter:%d] Loss: %.03f '
                    % (epoch + 1, (i + 1 + epoch * length), loss.item()))
            if args.debug:
                break

        # get the ac with testdataset in each epoch
        print('Waiting Test...')
        with torch.no_grad():
            correct = 0
            total = 0
            for data in val_loader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                total += labels.size(0)
                loss = angular_error(outputs, labels.float())
                # loss = criterion(yaw_pitch_to_vec(outputs), yaw_pitch_to_vec(labels.float()))
                correct += loss.item() * (labels.size(0))
                if args.debug:
                    break
            print('Test\'s loss is: %.03f' % (correct/total))

        filename = 'assets/model_saved/model_' + args.name + \
                   ':lr={lr},' \
                   'total_epoch={epoch},' \
                   'epoch_save={now},' \
                   'res_channels={res_channels}' \
                   '.pt'.format(
                       lr=LR, epoch=EPOCH, res_channels=res_channels, now=epoch
                   )
        torch.save(model.state_dict(), filename)

    print('Train has finished, total epoch is %d' % EPOCH)
    filename = 'assets/model_saved/model_' + args.name + \
               ':lr={lr},' \
               'total_epoch={epoch},' \
               'res_channels={res_channels}' \
               '.pt'.format(
                   lr=LR, epoch=EPOCH, res_channels=res_channels
               )
    torch.save(model.state_dict(), filename)


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
