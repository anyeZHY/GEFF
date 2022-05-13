import argparse
import math

import torch
import torch.nn as nn
import torch.optim as optim
from simclr.data_utils import make_transform
from ge.utils.dataloader import load_data
from ge.model.model_zoo import get_model, gen_geff
from simclr.contrastive_loss import simclr_loss

def train(args):
    print('SimCLR')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # ===== set hyperparameter >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    LR = args.lr
    print_every = args.print_every if (not args.debug) else 1
    EPOCH = args.epoch if (not args.debug) else 1
    BATCH_SIZE = args.batch if (not args.debug) else 16
    print(str(args)[10:-1])

    # prepare dataset and preprocessing
    T = make_transform(jitter=args.jitter, gray=args.gray)
    train_loader, val_loader = load_data(BATCH_SIZE, T)

    # ===== Model Configuration >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    dim_face = args.dim_face
    models = gen_geff(args,channels={'Face':dim_face, 'Out':out_channel}, device=device)
    model = get_model(args, models, args.useres).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_model = None
    best_loss = math.inf

    for epoch in range(0, EPOCH):
        print('\nEpoch: %d' % (epoch + 1))
        # ===== Training >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        model.train()
        length = len(train_loader)
        for i, data in enumerate(train_loader, 0):
            # prepare dataset
            imgs, labels = data
            imgs = {name: imgs[name].to(device) for name in imgs}
            labels = labels.to(device)
            optimizer.zero_grad()
            gaze = model(imgs)
            sim_loss = simclr_loss(gaze, labels, tau=0.5)
            loss = sim_loss
            loss.backward()
            optimizer.step()

            # print ac & loss in each batch
            if (i+1+epoch*length)%print_every == 0:
                print('[epoch:%d, iter:%d] Loss: %.03f '
                    % (epoch + 1, (i + 1 + epoch * length), loss.item()))

    print('Train has finished, total epoch is %d' % EPOCH)
    filename = str(args)[10:-1] + '.pt'
    print(filename)
    if not args.debug:
        torch.save(best_model, filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training Congfiguration')

    parser.add_argument("--debug", action="store_true", help="Train concisely and roughly")
    parser.add_argument("--save_every", action="store_true", help="Save models after every epoch")
    parser.add_argument("--out_channel", default=2, type=int)
    parser.add_argument("--print_every", default=50, type=int, help="Print loss")

    # hyperparameters in Trainnig part
    parser.add_argument("--model", default="baseline", choices=['baseline','fuse' ,'geff'] ,type=str)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--epoch", default=10, type=int)
    parser.add_argument("--batch", default=128, type=int)
    parser.add_argument("--dim_face", default=512, type=int)
    parser.add_argument("--weight", default=0.2, type=float, help="Weight in Vanilla Fusion model")
    parser.add_argument("--t", default=0.2, type=float, help="Weight in GEFF model")
    parser.add_argument("--jitter", default=0.2, type=float, help="Possibility of Jitter in data transformation")
    parser.add_argument("--gray", default=0.2, type=float, help="Possibility of converting the image into a Gray one")
    train(parser.parse_args())
