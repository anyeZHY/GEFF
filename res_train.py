# import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# import torchvision
from gaze_estimation.utils.dataloader import load_data_naive
from gaze_estimation.model.resnet import ResNet, ResBlock


# check gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# set hyperparameter
EPOCH = 10
pre_epoch = 0
BATCH_SIZE = 128
LR = 0.01
# out_channel = 2
# channels = (1, 1, 1, 1)

# prepare dataset and preprocessing
trainloader, testloader = load_data_naive(BATCH_SIZE)

# define ResNet18
net = ResNet(ResBlock).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters())

for epoch in range(pre_epoch, EPOCH):
    print('\nEpoch: %d' % (epoch + 1))
    net.train()
    sum_loss = 0.0
    correct = 0.0
    total = 0.0
    for i, data in enumerate(trainloader, 0):
        # prepare dataset
        length = len(trainloader)
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        # forward & backward
        outputs = net(inputs)
        loss = criterion(outputs, labels.float())
        # loss = F.binary_cross_entropy_with_logits(outputs, labels)
        # print(loss.item())
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
            loss = criterion(outputs, labels.float())
            correct += loss.item()
        print('Test\'s loss is: %.03f' % (correct))

print('Train has finished, total epoch is %d' % EPOCH)
