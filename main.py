import os
used_gpu = '0,1'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = used_gpu

import argparse
from utils import *
from Trainer import *
from torch.optim import *
from data_transform import *
from se_inception_v3 import *
from torch.utils.data.sampler import RandomSampler, SequentialSampler


def create_config():
    parser = argparse.ArgumentParser(description='Parameter22s for Cdiscount Classification')

    # Data Settings
    parser.add_argument('--train_bson_path', type=str, default='/data/lixiang/train.bson', help='where original training data')
    parser.add_argument('--num_classes', type=int, default=5270, help='how many classes to be classified')
    parser.add_argument('--num_train', type=int, default= 12371293, help='how many training datas')
    parser.add_argument('--data_woker', type=int, default=5, help='how many workers to read datas')

    # Model Settings
    parser.add_argument('--batch_size', type=int, default=2, help='how many samples in a batch')
    parser.add_argument('--image_size', type=tuple, default=(3,150,150), help='image size as (C, H, W)')
    parser.add_argument('--saved_model', type=str, default="epoch-14-acc-0.63.pkl",
                        help='the name of saved model')

    # Optimizer Settings
    parser.add_argument('--optimizer', type=str, default='SGD', help='which optimizer to apply')
    parser.add_argument('--initial_learning_rate', type=float, default=0.0001, help='initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay')

    return parser.parse_args().__dict__


num_classes = 5270
# total = 7069896
a = "LB=0.69673_se-inc3_00026000_model.pth"
cuda = torch.cuda.is_available()
total = 12371293
# total = 82
np.random.seed(2333)
to = np.arange(total)
to = np.random.permutation(to)
val_mask = to[:int(total*0.1)]
train_mask = to[int(total*0.1):]
print("finish mask")


def run(cfg):
    net = SEInception3(in_shape=cfg["image_size"], num_classes=cfg["num_classes"])
    if len(used_gpu) > 1 and cuda:
        net = torch.nn.DataParallel(net, device_ids=[0,1])

    if cfg["saved_model"]:
        print("*-------Begin Loading Saved Models!------*")
        if len(used_gpu) > 1 and cuda:
            net.module.load_pretrained_model('saved_models/' + cfg["saved_model"])
        else:
            net.load_pretrained_model('saved_models/' + cfg["saved_model"])

    if cfg['optimizer'] == 'SGD':
        optimizer = SGD(filter(lambda p: p.requires_grad, net.parameters()),
                        lr=cfg['initial_learning_rate'], momentum=cfg['momentum'],
                        weight_decay=cfg['weight_decay'])
    elif cfg['optimizer'] == 'Adam':
        optimizer = Adam(filter(lambda p: p.requires_grad, net.parameters()),
                         lr=cfg['initial_learning_rate'],
                         weight_decay=cfg['weight_decay'])

    loss = F.cross_entropy
    trainer = Trainer(net, optimizer, loss, cfg['batch_size'])
    lr_step = MultiStepLR(optimizer, [3, 6])
    # lr_step = ReduceLROnPlateau(optimizer, 'min', patience=3)

    print("*----------Begin Loading Data!-----------*")
    data_frame = extract_categories_df(cfg['train_bson_path'])
    train_dataset = CdiscountTrain(cfg['train_bson_path'], data_frame, train_mask, transform=train_augment)
    train_loader = DataLoader(train_dataset,
                              sampler=RandomSampler(train_dataset),
                              batch_size=cfg['batch_size'],
                              drop_last=True,
                              num_workers=cfg['data_woker'])

    valid_dataset = CdiscountTest(cfg['train_bson_path'], data_frame, val_mask, transform=valid_augment)
    valid_loader = DataLoader(valid_dataset,
                              sampler=SequentialSampler(valid_dataset),
                              batch_size=cfg['batch_size'],
                              drop_last=False,
                              num_workers=cfg['data_woker'])

    print("*------------Begin Training!-------------*")
    trainer.loop(train_loader, valid_loader, lr_step)


if __name__ == "__main__":
    cfg =create_config()
    run(cfg)
