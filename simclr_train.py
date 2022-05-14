import argparse
import math

import torch
import torch.nn as nn
import torch.optim as optim
from simclr.data_utils import load_data_sim
from simclr.contrastive_loss import simclr_loss, simclr_fe
from simclr.model import SimCLR

def train(args):
    print('SimCLR')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # ===== set hyperparameter >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    LR = args.lr
    EPOCH = args.epoch if (not args.debug) else 1
    BATCH_SIZE = args.batch if (not args.debug) else 16
    print(str(args)[10:-1])

    # prepare dataset and preprocessing
    train_loader = load_data_sim(args, BATCH_SIZE)
    model = SimCLR()
    model.train()

    # ===== Model Configuration >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(0, EPOCH):
        print('\nEpoch: %d' % (epoch + 1))
        # ===== Training >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        length = len(train_loader)
        for i, data in enumerate(train_loader, 0):
            # prepare dataset
            imgs_i, imgs_j = data
            imgs_i = {name: imgs_i[name].to(device) for name in imgs_i}
            imgs_j = {name: imgs_i[name].to(device) for name in imgs_j}
            optimizer.zero_grad()
            f_i, l_i, r_i = model(imgs_i)
            f_j, l_j, r_j = model(imgs_j)
            sim_loss_face = simclr_loss(f_i, f_j, tau=0.5, device=device)
            sim_loss_left = simclr_loss(l_i, l_j, tau=0.5, device=device)
            sim_loss_right = simclr_loss(r_i, r_j, tau=0.5, device=device)
            sim_loss_fe = simclr_fe(f_i, l_i, r_i, f_j, l_j, r_j, tau=0.5)
            loss = sim_loss_face + sim_loss_left + sim_loss_right + 0.5 * sim_loss_fe
            loss.backward()
            optimizer.step()

            # print ac & loss in each batch
            print('[epoch:%d, iter:%d] Loss: %.03f '% (epoch + 1, (i + 1 + epoch * length), loss.item()))

    print('Train has finished, total epoch is %d' % EPOCH)
    filename = 'assets/model_saved/simclr.pt'
    print(filename)
    torch.save(model, filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training Congfiguration')

    parser.add_argument("--debug", action="store_true", help="Train concisely and roughly")

    # hyperparameters in Trainnig part
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--epoch", default=100, type=int)
    parser.add_argument("--batch", default=1024, type=int)
    parser.add_argument("--jitter", default=0.6, type=float, help="Possibility of Jitter in data transformation")
    parser.add_argument("--gray", default=0.2, type=float, help="Possibility of converting the image into a Gray one")
    train(parser.parse_args())
