import argparse
import math

import torch
import torch.nn as nn
import torch.optim as optim
from simclr.data_utils import make_transform
from gaze.utils.dataloader import load_data
from gaze.utils.make_loss import angular_error
from gaze.model.model_zoo import get_model, gen_geff


def train(args, person_id=9, device='cuda'):

    # ===== set hyperparameter >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    print_every = args.print_every if (not args.debug) else 1
    EPOCH = args.epoch if (not args.debug) else 1
    BATCH_SIZE = args.batch if (not args.debug) else 2
    LR = args.lr
    out_channel = args.out_channel

    # prepare dataset and preprocessing
    T = make_transform(jitter=args.jitter, gray=args.gray, blur=0, sharp=0, posterize=0) if args.data_aug else None
    train_loader, val_loader = load_data(
        args, BATCH_SIZE,
        transform_train=T,
        val_size=BATCH_SIZE,
        flip=args.flip,
        person_id=person_id
    )

    # ===== Model Configuration >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    dim_face = args.dim_face
    dim_eyes = dim_face//4
    usebn = args.usebn or args.name == 'simclr'
    models = gen_geff(
        args, device=device,
        channels={'Face':dim_face,'Out':out_channel,'Fusion':[dim_face + dim_eyes, 1]},
        idx=person_id
    )
    model = get_model(args, models).to(device)
    # if args.debug:
    #     print(model.face_en.state_dict().items())
    L1 = nn.SmoothL1Loss(reduction='mean') if args.loss=='L1' else nn.MSELoss(reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=LR)
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma)

    best_model = None
    best_loss = math.inf
    loss_log = []

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
            gaze = model(imgs, args.pretrain) if args.model!='geff' \
                else model(imgs, args.model, args.pretrain, args.warm, cur_epoch=epoch, usebn=usebn)
            ang_loss = angular_error(gaze, labels)
            L1_loss = L1(gaze, labels.float())
            loss = L1_loss
            loss.backward()
            optimizer.step()

            # print ac & loss in each batch
            if (i+1+epoch*length) % print_every == 0:
                print('[epoch:%d, iter:%d] Loss: %.05f AngularLoss: %.05f'
                    % (epoch + 1, (i + 1 + epoch * length), L1_loss.item(), ang_loss.item()))
            if args.debug:
                break
        # scheduler.step()
        # ===== Evaluation >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        print('Waiting Test...')
        model.eval()
        aug_loss = 0
        L1_loss = 0
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
                aug_loss += loss.item() * (labels.size(0))
                L1_loss += L1(gaze, labels.float()) * (labels.size(0))
                if args.debug:
                    break
        print('Test\'s AngularLoss is: %.03f' % (aug_loss/total))
        loss_log.append(aug_loss/total)
        if best_loss > aug_loss/total:
            best_loss = aug_loss / total
            best_model = model

        # if args.debug:
        #     break
        filename = 'assets/model_saved/mid:' + args.model + args.name + '.pt'
        if not args.debug and epoch == EPOCH//2:
            torch.save(best_model, filename)

    print('Train has finished, total epoch is %d' % EPOCH)
    if args.model == 'baseline':
        filename = 'assets/model_saved/{}'.format(person_id) + args.model + args.name + '.pt'
    else:
        filename = 'assets/model_saved/' + args.model + args.name + '.pt'
    filename_final = 'assets/model_saved/final{}' + args.model + args.name + '.pt'
    print(filename)
    print('test loss:', loss_log)
    print('best loss:', best_loss)
    if not args.debug:
        torch.save(best_model, filename)
        torch.save(model, filename_final)
    return best_loss


def make_parser():
    parser = argparse.ArgumentParser(description='Training Congfiguration')

    parser.add_argument("--name", default="Name", type=str, help='Config File names')
    parser.add_argument("--debug", action="store_true", help="Train concisely and roughly")
    parser.add_argument("--save_every", action="store_true", help="Save models after every epoch")
    parser.add_argument("--print_every", default=50, type=int, help="Print loss")
    parser.add_argument("--out_channel", default=2, type=int)

    # hyperparameters in Trainnig part
    parser.add_argument("--model", default="baseline", choices=['baseline', 'fuse', 'geff', 'febase', 'simclr'],
                        type=str)
    parser.add_argument("--dataset", default="mpii", choices=['mpii', 'columbia'], type=str)
    parser.add_argument("--cross_val_off", action="store_true", help="Cross Validation off")
    parser.add_argument("--warm", default=30, type=int)
    parser.add_argument("--epoch", default=20, type=int)
    parser.add_argument("--batch", default=128, type=int)
    parser.add_argument("--loss", default='L1', type=str)
    parser.add_argument("--lr", default=0.001, type=float)

    parser.add_argument("--dim_face", default=512, type=int)
    parser.add_argument("--weight", default=0.2, type=float, help="Weight in Vanilla Fusion model")
    parser.add_argument("--t", default=1, type=float, help="Weight in GEFF model")
    parser.add_argument("--eye_en", choices=['resnet', 'mlp', 'conv'])
    parser.add_argument("--usebn", action="store_true", help="Use BatchNorm in GEFF")
    parser.add_argument("--pretrain", action="store_true")

    parser.add_argument("--data_aug", action="store_true", help="Augment data")
    parser.add_argument("--mask", action="store_true", help="Use masks on eyes")
    parser.add_argument("--jitter", default=0.2, type=float, help="Possibility of Jitter in data transformation")
    parser.add_argument("--gray", default=0.2, type=float, help="Possibility of converting the image into a Gray one")
    parser.add_argument("--flip", default=0.0, type=float)
    return parser


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    parser = make_parser()
    print(str(parser.parse_args())[10:-1])
    data_set = parser.parse_args().dataset
    losses = []
    args = parser.parse_args()
    if data_set == 'mpii':
        if args.cross_val_off:
            losses.append(train(args, 9, device))
        else:
            for i in range(10):
                print('\n===== person\'s ID: {} >>>>>>'.format(i))
                losses.append(train(args, device=device, person_id=i))
    else:
        losses.append(train(args, 42, device))
    print(losses)
    print('AvgLoss:', torch.mean(torch.Tensor(losses)).item())
