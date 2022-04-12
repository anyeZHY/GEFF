# import pandas as pd
import argparse
import torch
# import torch.nn as nn
import torch.optim as optim
# import torch.nn.functional as F
# import torchvision
from gaze_estimation.utils.dataloader import load_data_naive
from gaze_estimation.utils.make_loss import angular_error
from gaze_estimation.model.resnet import ResNet, ResBlock

def train(args):
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
    trainloader, testloader = load_data_naive(BATCH_SIZE)

    # define ResNet18
    net = ResNet(ResBlock, out_channel=out_channel, channels=res_channels).to(device)

    # criterion = nn.SmoothL1Loss(reduction='mean')
    optimizer = optim.Adam(net.parameters(), lr=LR)

    for epoch in range(pre_epoch, EPOCH):
        print('\nEpoch: %d' % (epoch + 1))
        net.train()
        sum_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # prepare dataset
            length = len(trainloader)
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            # forward & backward
            outputs = net(inputs)
            loss = angular_error(outputs, labels.float())
            # loss = criterion(yaw_pitch_to_vec(outputs), yaw_pitch_to_vec(labels.float()))
            # loss = torch.mean(angular_error(outputs, labels.float()))
            loss.backward()
            optimizer.step()

            # print ac & loss in each batch
            sum_loss += loss.item()
            print('[epoch:%d, iter:%d] Loss: %.03f '
                  % (epoch + 1, (i + 1 + epoch * length), loss.item()))

        # get the ac with testdataset in each epoch
        print('Waiting Test...')
        with torch.no_grad():
            correct = 0
            total = 0
            for data in testloader:
                net.eval()
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                total += labels.size(0)
                loss = angular_error(outputs, labels.float())
                # loss = criterion(yaw_pitch_to_vec(outputs), yaw_pitch_to_vec(labels.float()))
                correct += loss.item()
            print('Test\'s loss is: %.03f' % correct)

    print('Train has finished, total epoch is %d' % EPOCH)
    filename = 'assets/model' + args.name +\
               ':lr={lr},' \
               'epoch={epoch},' \
               'res_channels={res_channels}' \
               '.pt'.format(
                   lr=LR, epoch=EPOCH, res_channels=res_channels
               )
    torch.save(net.state_dict(), filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training Congfiguration')
    parser.add_argument("--epoch", default=10, type=int)
    parser.add_argument("--batch", default=128, type=int)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--out_channel", default=2, type=int)
    parser.add_argument("--res_channels", nargs='+', default=[16, 32, 64, 128], type=int)
    parser.add_argument("--name", default="_", type=str, help='The file name to save model')
    train(parser.parse_args())
