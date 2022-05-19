# To Do:
# 1. Get angular loss on our test set
# 2. Provide a method to compute the angular loss on ungiven datas
# 3. Compute angluar loss on another dataset (the domain of labels is quite differnent)
import sys
from os.path import dirname, abspath
path = dirname(dirname(abspath(__file__)))
sys.path.append(path)
import torch
from torchvision import transforms
from gaze.utils.dataloader import MPII, make_transform
from torch.utils.data import DataLoader
from gaze.utils.make_loss import angular_error
def get_test(BATCH_SIZE):
    # df_data = procees_data(0)
    # df_data = pd.read_pickle('assets/MPII_2D_annoataion.csv')
    train_file = 'assets/MPII_test.csv'
    img_dir = 'assets/MPIIFaceGaze/Image'

    transform_eye, transform_val = make_transform()
    test_set = MPII(train_file, img_dir, transform_val, transform_eye)

    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)
    return test_loader


def test(path='assets/model_saved/MPII/geffMFP.pt'):
    print(path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 100
    test_loader = get_test(BATCH_SIZE)
    model = torch.load(path, map_location=torch.device(device))
    print('Waiting Test...')
    sum_loss = 0
    total = 0
    for data in test_loader:
        imgs, labels = data
        imgs = {name: imgs[name].to(device) for name in imgs}
        labels = labels.to(device)
        gaze = model(imgs)
        total += labels.size(0)
        loss = angular_error(gaze, labels.float())
        sum_loss += loss.item() * (labels.size(0))
    print('Test\'s loss is: %.03f' % (sum_loss/total))

test()
test('assets/model_saved/MPII/fuseMFP.pt')
test('assets/model_saved/MPII/BaseLr.pt')
