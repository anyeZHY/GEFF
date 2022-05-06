import argparse
import math

import torch
import torch.nn as nn
import torch.optim as optim
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

    # ===== set hyperparameter >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    print_every = args.print_every if (not args.debug) else 1
    EPOCH = args.epoch if (not args.debug) else 1
    BATCH_SIZE = args.batch if (not args.debug) else 16
    LR = args.lr
    out_channel = args.out_channel
    print('lr={lr},total_epoch={epoch}'.format(lr=LR, epoch=EPOCH))

    # prepare dataset and preprocessing
    train_loader, val_loader = load_data(BATCH_SIZE)

    # ===== Model Configuration >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    dim_face = args.dim_face
    models = gen_geff(args,channels={'Face':dim_face, 'Out':out_channel}, device=device)
    model = get_model(args, models)
    L1 = nn.SmoothL1Loss(reduction='mean')
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
            optimizer.zero_grad()
            gaze = model(imgs)
            ang_loss = angular_error(gaze, labels)
            L1_loss = L1(gaze, labels.float())
            loss = ang_loss + 0.5 * L1_loss
            loss.backward()
            optimizer.step()

            # print ac & loss in each batch
            if (i+1+epoch*length)%print_every == 0:
                print('[epoch:%d, iter:%d] Loss: %.03f '
                    % (epoch + 1, (i + 1 + epoch * length), loss.item()))
            if args.debug:
                break
        # ===== Evaluation >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        print('Waiting Test...')
        model.eval()
        sum_loss = 0
        total = 0
        with torch.no_grad():
            for data in val_loader:
                imgs, labels = data
                gaze = model(imgs)
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


        if args.debug:
            break
        filename = 'assets/model_saved/' + args.model + \
                   ',lr={lr},' \
                   'epoch_save={now},' \
                   '.pt'.format(
                       lr=LR, now=epoch+1
                   )
        if args.save_every:
            torch.save(model.state_dict(), filename)

    print('Train has finished, total epoch is %d' % EPOCH)
    filename = 'assets/model_saved/' + args.model + ',lr={lr}.pt'.format(lr=LR)
    if not args.debug:
        torch.save(best_model.state_dict(), filename)


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
    parser.add_argument("--dim_face", default=512, type=int)
    parser.add_argument("--res_channels", nargs='+', default=[16, 32, 64, 128], type=int)
    parser.add_argument("--model", default="baseline", choices=['baseline','fuse' ,'geff'] ,type=str)
    train(parser.parse_args())
