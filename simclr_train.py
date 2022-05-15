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
    EPOCH = args.epoch if (not args.debug) else 100
    BATCH_SIZE = args.batch if (not args.debug) else 2
    tau = args.tau
    print(str(args)[10:-1])

    # prepare dataset and preprocessing
    train_loader = load_data_sim(args, BATCH_SIZE)
    model = SimCLR()
    if args.multi_gpu:
        model = nn.DataParallel(model)
    model = model.to(device)
    model.train()

    # ===== Model Configuration >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(0, EPOCH):
        print('\nEpoch: %d' % (epoch + 1))
        # ===== Training >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        length = len(train_loader)
        for i, data in enumerate(train_loader, 0):
            # prepare dataset
            imgs_i, imgs_j= data
            imgs_i = {name: imgs_i[name].to(device) for name in imgs_i}
            imgs_j = {name: imgs_j[name].to(device) for name in imgs_j}
            optimizer.zero_grad()
            f_i, l_i, r_i = model(imgs_i)
            f_j, l_j, r_j = model(imgs_j)
            _, D = l_i.shape
            loss_face = simclr_loss(f_i[:,2*D:], f_j[:,2*D:], tau=tau, device=device)
            loss_left = simclr_loss(l_i, l_j, tau=tau, device=device)
            loss_right = simclr_loss(r_i, r_j, tau=tau, device=device)
            loss_fe = simclr_fe(f_i, l_i, r_i, f_j, l_j, r_j, tau=tau)
            loss = loss_face + loss_left + loss_right + 0.5 * loss_fe
            loss.backward()
            optimizer.step()

            # print ac & loss in each batch
            if i%20==0:
                print('[epoch:%d, iter:%d] Loss: %.03f '% (epoch + 1, (i + 1 + epoch * length), loss.item()))
            if args.debug:
                print('face: %3f, left: %3f, right: %3f, fe: %3f'
                      % (loss_face.item(), loss_left.item(), loss_right.item(), loss_fe.item()))
                break
        if (epoch+1)%100==0:
            filename = 'assets/model_saved/' + args.name + 'simclr{}.pt'.format(epoch+1)
            print(filename)
            if args.multi_gpu:
                torch.save(model.module, filename)
            else:
                torch.save(model, filename)
    print('Train has finished, total epoch is %d' % EPOCH)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training Congfiguration')

    parser.add_argument("--debug", action="store_true", help="Train concisely and roughly")
    parser.add_argument("--multi_gpu", action="store_true")
    parser.add_argument("--name", default="", type=str)

    # hyperparameters in Trainnig part
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--epoch", default=100, type=int)
    parser.add_argument("--batch", default=1024, type=int)
    parser.add_argument("--jitter", default=0.5, type=float, help="Possibility of Jitter in data transformation")
    parser.add_argument("--gray", default=0.2, type=float, help="Possibility of converting the image into a Gray one")
    parser.add_argument("--blur", default=0.2, type=float)
    parser.add_argument("--sharp", default=0.2, type=float)
    parser.add_argument("--posterize", default=0.2, type=float)
    parser.add_argument("--tau", default=0.5, type=float)
    train(parser.parse_args())
