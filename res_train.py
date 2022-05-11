import argparse
import math

import torch
import torch.nn as nn
import torch.optim as optim
from simclr.data_utils import make_transform
from ge.utils.dataloader import load_data
from ge.utils.make_loss import angular_error
from ge.model.model_zoo import get_model, gen_geff

def train(args):
    """
    Input
    - arg: Arguments of:
        - epoch: Default = 10
        - lr: Learning rate. Default = 1e-3
        - out_channel: The size of output. Default = 2
        - res_channels: The channels of ResNet, of shape (4, ). Default = (16, 32, 64, 128)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # ===== set hyperparameter >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    print_every = args.print_every if (not args.debug) else 1
    EPOCH = args.epoch if (not args.debug) else 5
    BATCH_SIZE = args.batch if (not args.debug) else 16
    LR = args.lr
    out_channel = args.out_channel
    print(str(args)[10:-1])

    # prepare dataset and preprocessing
    T = make_transform(jitter=args.jitter, gray=args.gray) if args.data_aug else None
    train_loader, val_loader = load_data(BATCH_SIZE, transform_train=T, val_size=100, flip=args.flip)

    # ===== Model Configuration >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    dim_face = args.dim_face
    dim_eyes = dim_face//4
    models = gen_geff(
        args, device=device,
        channels={'Face':dim_face,'Out':out_channel,'Fusion':[2*dim_eyes, 1]}
    )
    model = get_model(args, models, args.useres).to(device)
    # if args.debug:
    #     print(model.face_en.state_dict().items())
    L1 = nn.SmoothL1Loss(reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)

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
            gaze = model(imgs, args)
            ang_loss = angular_error(gaze, labels)
            L1_loss = L1(gaze, labels.float())
            loss = ang_loss + 10 * L1_loss
            loss.backward()
            optimizer.step()

            # print ac & loss in each batch
            if (i+1+epoch*length)%print_every == 0:
                print('[epoch:%d, iter:%d] Loss: %.03f '
                    % (epoch + 1, (i + 1 + epoch * length), loss.item()))
            if args.debug:
                break
        scheduler.step()
        # ===== Evaluation >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        print('Waiting Test...')
        model.eval()
        sum_loss = 0
        total = 0
        with torch.no_grad():
            for data in val_loader:
                imgs, labels = data
                imgs = {name: imgs[name].to(device) for name in imgs}
                labels = labels.to(device)
                gaze = model(imgs, args)
                total += labels.size(0)

                loss = angular_error(gaze, labels.float())
                # loss = criterion(yaw_pitch_to_vec(F_face), yaw_pitch_to_vec(labels.float()))
                sum_loss += loss.item() * (labels.size(0))
                if args.debug:
                    break
        print('Test\'s loss is: %.03f' % (sum_loss/total))
        if best_loss > sum_loss/total:
            best_loss = sum_loss / total
            best_model = model

        filename = 'assets/model_saved/' + str(args)[10:-1] + 'epoch_save={now}.pt'.format(now=epoch+1)
        if args.save_every:
            torch.save(model.state_dict(), filename)
        # if args.debug:
        #     break

    print('Train has finished, total epoch is %d' % EPOCH)
    filename = 'assets/model_saved/' + str(args)[10:-1] + '.pt'
    print(filename)
    if not args.debug:
        torch.save(best_model, filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training Congfiguration')

    parser.add_argument("--name", default="Name", type=str, help='Config File names')
    parser.add_argument("--debug", action="store_true", help="Train concisely and roughly")
    parser.add_argument("--save_every", action="store_true", help="Save models after every epoch")
    parser.add_argument("--print_every", default=50, type=int, help="Print loss")
    parser.add_argument("--out_channel", default=2, type=int)

    # hyperparameters in Trainnig part
    parser.add_argument("--model", default="baseline", choices=['baseline','fuse' ,'geff'] ,type=str)
    parser.add_argument("--pretrain", action="store_true")
    parser.add_argument("--epoch", default=10, type=int)
    parser.add_argument("--batch", default=128, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--dim_face", default=512, type=int)
    parser.add_argument("--weight", default=0.2, type=float, help="Weight in Vanilla Fusion model")
    parser.add_argument("--t", default=0.2, type=float, help="Weight in GEFF model")
    parser.add_argument("--useres", action="store_true", help="Use resnet as eyes' encoder")
    parser.add_argument("--data_aug", action="store_true", help="Augment data")
    parser.add_argument("--jitter", default=0.2, type=float, help="Possibility of Jitter in data transformation")
    parser.add_argument("--gray", default=0.2, type=float, help="Possibility of converting the image into a Gray one")
    parser.add_argument("--lr_step", default=500, type=int)
    parser.add_argument("--lr_gamma", default=0.5, type=float)
    parser.add_argument("--flip", default=0.0, type=float)
    train(parser.parse_args())
