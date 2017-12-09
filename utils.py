import os
import cv2
import bson
import torch
import struct
import numpy as np
import pandas as pd
from tqdm import tqdm
from bisect import bisect_right
from collections import defaultdict
from data_transform import image_to_tensor
from torch.optim import Optimizer
from torchvision import transforms as transf
from torch.utils.data import Dataset, DataLoader
from label_id_dict import label_to_category_id, category_id_to_label


# ====================================================================================== #
# get train datas by extracting all entries
# ====================================================================================== #
def read_bson(bson_path, num_records, with_categories):
    rows = {}
    imgs_to_choice_num = {1:1, 2:2, 3:6, 4:6}
    with open(bson_path, "rb") as f, tqdm(total=num_records) as pbar:
        offset = 0
        records_read = 0
        while True:
            item_length_bytes = f.read(4)
            if len(item_length_bytes) == 0:
                break

            length = struct.unpack("<i", item_length_bytes)[0]
            f.seek(offset)
            item_data = f.read(length)
            assert len(item_data) == length

            item = bson.BSON.decode(item_data)
            product_id = item["_id"]
            num_imgs = len(item["imgs"])

            row = [num_imgs, offset, length]
            if with_categories:
                row += [item["category_id"]]
                row += [np.random.choice(imgs_to_choice_num[num_imgs])]
            rows[product_id] = row

            offset += length
            f.seek(offset)
            records_read += 1
            pbar.update()
    pbar.close()
    columns = ["num_imgs", "offset", "length"]
    if with_categories:
        columns += ["category_id", "choice"]

    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "product_id"
    df.columns = columns
    df.sort_index(inplace=True)
    return df


def get_data_frame(data_path, n_train, is_train):
    cat2idx, idx2cat = category_id_to_label, label_to_category_id
    data_frame = read_bson(data_path, n_train, with_categories=is_train)
    if is_train:
        data_frame.category_id = np.array([cat2idx[ind] for ind in data_frame.category_id])
    return data_frame


def get_obs(fname, offset, length):
    fobj = open(fname, 'rb')
    fobj.seek(offset)
    res = bson.BSON.decode(fobj.read(length))
    fobj.close()
    return res


tuple2 = [[[0], [1]], [[1], [0]]]


tuple3 = [[[0], [1,2]], [[1], [0,2]], [[2], [0,1]],
          [[1,2], [0]], [[0,2], [1]], [[0,1], [2]]]


tuple4 = [[[1,2], [3,0]], [[1,3],[2,0]], [[1,0], [2,3]],
          [[2,3], [1,0]], [[2,0],[1,3]], [[3,0], [1,2]]]


def get_choice_set(num_img):
    if num_img == 1:
        return [[[0], [0]]]
    elif num_img == 2:
        return tuple2
    elif num_img == 3:
        return tuple3
    elif num_img == 4:
        return tuple4


class CdiscountTrainDataset(Dataset):
    def __init__(self, data_path, data_frame, transform):
        self.data_path = data_path
        self.data_frame = data_frame
        self.transform = transform

    def __len__(self):
        return self.data_frame.index.values.shape[0]

    def __getitem__(self, index):
        entry = self.data_frame.iloc[index]
        num_imgs, offset, length, target, choice = entry
        obs = get_obs(self.data_path, offset, length)

        keep_set = get_choice_set(num_imgs)[choice][0]
        keep = keep_set[np.random.choice(len(keep_set))]

        byte_str = obs['imgs'][keep]['picture']
        img = cv2.imdecode(np.fromstring(byte_str, dtype=np.uint8), cv2.IMREAD_COLOR)
        img = self.transform(img)
        return img, target


class CdiscountValDataset(Dataset):
    def __init__(self, data_path, data_frame, transform):
        self.data_path = data_path
        self.data_frame = data_frame
        self.transform = transform

    def __len__(self):
        return self.data_frame.index.values.shape[0]

    def __getitem__(self, index):
        entry = self.data_frame.iloc[index]
        num_imgs, offset, length, target, choice = entry
        obs = get_obs(self.data_path, offset, length)

        keep_set = get_choice_set(num_imgs)[choice][1]
        keep = keep_set[np.random.choice(len(keep_set))]

        byte_str = obs['imgs'][keep]['picture']
        img = cv2.imdecode(np.fromstring(byte_str, dtype=np.uint8), cv2.IMREAD_COLOR)
        img = self.transform(img)
        return img, target


class CdiscountTestDataset(Dataset):
    def __init__(self, data_path, data_frame, transform=None):
        self.data_path = data_path
        self.data_frame = data_frame
        self.transform = transform

    def __len__(self):
        return self.data_frame.index.values.shape[0]

    def __getitem__(self, index):
        entry = self.data_frame.iloc[index]
        num_imgs, offset, length = entry
        obs = get_obs(self.data_path, offset, length)
        keep = np.random.choice(len(obs['imgs']))
        byte_str = obs['imgs'][keep]['picture']
        img = cv2.imdecode(np.fromstring(byte_str, dtype=np.uint8), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(img) if self.transform else image_to_tensor(img)
        return img


# ====================================================================================== #
# get train datas by extracting all images
# ====================================================================================== #
def extract_categories_df(bson_path, num_images=None):
    if not num_images and os.path.exists("all_images_categories.csv"):
        print("loading from csv file: all_images_categories.csv")
        return pd.read_csv("all_images_categories.csv")
    elif num_images and os.path.exists('{}_images_categories.csv'.format(num_images)):
        print("loading from csv file: {}_images_categories.csv".format(num_images))
        return pd.read_csv("{}_images_categories.cvs".format(num_images))
    print("loading from bson file: {}".format(bson_path))
    img_category = list()
    item_locs_list = list()
    items_len_list = list()
    pic_ind_list = list()

    with open(bson_path, 'rb') as f:
        data = bson.decode_file_iter(f)
        last_item_loc = 0
        for c, d in enumerate(data):
            loc = f.tell()
            item_len = loc - last_item_loc
            category_id = d['category_id']

            for e, pic in enumerate(d['imgs']):

                img_category.append(category_id)
                item_locs_list.append(last_item_loc)
                items_len_list.append(item_len)
                pic_ind_list.append(e)

                if num_images is not None:
                    if len(img_category) >= num_images:
                        break

            last_item_loc = loc

            if num_images is not None:
                if len(img_category) >= num_images:
                    break
    f.close()
    df_dict = {
        'category': img_category,
        "img_id": range(len(img_category)),
        "item_loc": item_locs_list,
        "item_len": items_len_list,
        "pic_ind": pic_ind_list
    }
    df = pd.DataFrame(df_dict)
    if not num_images:
        df.to_csv("all_images_categories.csv", index=False, sep=",")
    else:
        df.to_csv("{}_images_categories.csv".format(num_images), index=False, sep=",")
    return df


class CdiscountTrain(Dataset):
    def __init__(self, data_path, dataframe, train_mask, transform):
        self.data_path = data_path
        self.dataframe = dataframe
        self.mask = train_mask
        self.transform = transform

    def __getitem__(self, index):
        entry = self.dataframe.iloc[self.mask[index]]
        category_id, img_id, item_len, item_loc, pic_ind = entry
        obs = get_obs(self.data_path, item_loc, item_len)
        byte_str = obs['imgs'][pic_ind]['picture']
        img = cv2.imdecode(np.fromstring(byte_str, dtype=np.uint8), cv2.IMREAD_COLOR)
        img = self.transform(img)
        return img, category_id_to_label[category_id]

    def __len__(self):
        return len(self.mask)


class CdiscountVal(Dataset):
    def __init__(self, data_path, dataframe, val_mask, transform):
        self.data_path = data_path
        self.dataframe = dataframe
        self.mask = val_mask
        self.transform = transform

    def __getitem__(self, index):
        entry = self.dataframe.iloc[self.mask[index]]
        category_id, img_id, item_len, item_loc, pic_ind = entry
        obs = get_obs(self.data_path, item_loc, item_len)
        byte_str = obs['imgs'][pic_ind]['picture']
        img = cv2.imdecode(np.fromstring(byte_str, dtype=np.uint8), cv2.IMREAD_COLOR)
        img = self.transform(img)
        return img, category_id_to_label[category_id]

    def __len__(self):
        return len(self.mask)


# ====================================================================================== #
# learning rate adjustment, copied from pytorch's master
# ====================================================================================== #
class _LRScheduler(object):
    def __init__(self, optimizer, last_epoch=-1):
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def get_lr(self):
        raise NotImplementedError

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


class MultiStepLR(_LRScheduler):
    def __init__(self, optimizer, milestones, gamma=0.1, last_epoch=-1):
        if not list(milestones) == sorted(milestones):
            raise ValueError('Milestones should be a list of'
                             ' increasing integers. Got {}', milestones)
        self.milestones = milestones
        self.gamma = gamma
        super(MultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * self.gamma ** bisect_right(self.milestones, self.last_epoch)
                for base_lr in self.base_lrs]


class StepLR(_LRScheduler):
    def __init__(self, optimizer, step_size, gamma=0.1, last_epoch=-1):
        self.step_size = step_size
        self.gamma = gamma
        super(StepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * self.gamma ** (self.last_epoch // self.step_size)
                for base_lr in self.base_lrs]


class ReduceLROnPlateau(object):
    def __init__(self, optimizer, mode='min', factor=0.1, patience=10,
                 verbose=False, threshold=1e-4, threshold_mode='rel',
                 cooldown=0, min_lr=0, eps=1e-8):

        if factor >= 1.0:
            raise ValueError('Factor should be < 1.0.')
        self.factor = factor

        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        if isinstance(min_lr, list) or isinstance(min_lr, tuple):
            if len(min_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} min_lrs, got {}".format(
                    len(optimizer.param_groups), len(min_lr)))
            self.min_lrs = list(min_lr)
        else:
            self.min_lrs = [min_lr] * len(optimizer.param_groups)

        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.best = None
        self.num_bad_epochs = None
        self.mode_worse = None  # the worse value for the chosen mode
        self.is_better = None
        self.eps = eps
        self.last_epoch = -1
        self._init_is_better(mode=mode, threshold=threshold,
                             threshold_mode=threshold_mode)
        self._reset()

    def _reset(self):
        """Resets num_bad_epochs counter and cooldown counter."""
        self.best = self.mode_worse
        self.cooldown_counter = 0
        self.num_bad_epochs = 0

    def step(self, metrics, epoch=None):
        current = metrics
        if epoch is None:
            epoch = self.last_epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

        if self.num_bad_epochs > self.patience:
            self._reduce_lr(epoch)
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0

    def _reduce_lr(self, epoch):
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = max(old_lr * self.factor, self.min_lrs[i])
            if old_lr - new_lr > self.eps:
                param_group['lr'] = new_lr
                if self.verbose:
                    print('Epoch {:5d}: reducing learning rate'
                          ' of group {} to {:.4e}.'.format(epoch, i, new_lr))

    @property
    def in_cooldown(self):
        return self.cooldown_counter > 0

    def _init_is_better(self, mode, threshold, threshold_mode):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if threshold_mode not in {'rel', 'abs'}:
            raise ValueError('threshold mode ' + mode + ' is unknown!')
        if mode == 'min' and threshold_mode == 'rel':
            rel_epsilon = 1. - threshold
            self.is_better = lambda a, best: a < best * rel_epsilon
            self.mode_worse = float('Inf')
        elif mode == 'min' and threshold_mode == 'abs':
            self.is_better = lambda a, best: a < best - threshold
            self.mode_worse = float('Inf')
        elif mode == 'max' and threshold_mode == 'rel':
            rel_epsilon = threshold + 1.
            self.is_better = lambda a, best: a > best * rel_epsilon
            self.mode_worse = -float('Inf')
        else:  # mode == 'max' and epsilon_mode == 'abs':
            self.is_better = lambda a, best: a > best + threshold
            self.mode_worse = -float('Inf')


# ====================================================================================== #
# Other useful functions
# ====================================================================================== #
def get_state_dict(file):
    try:
        pretrain_state_dict = torch.load(file)
    except AssertionError:
        pretrain_state_dict = torch.load(file, map_location=lambda storage, location: storage)
    return pretrain_state_dict


def load_optimizer(optimizer, pretrained_optimizer_file):
    pretrain_state_dict = get_state_dict(pretrained_optimizer_file)
    optimizer.load_state_dict(pretrain_state_dict)
    optimizer.state = defaultdict(dict, optimizer.state)
    return


def filtered_params(net, param_list=None):
    def in_param_list(s):
        for p in param_list:
            if s.endswith(p):
                return True
        return False
    # Caution: DataParallel prefixes '.module' to every parameter name
    params = net.named_parameters() if param_list is None \
        else (p for p in net.named_parameters() if in_param_list(p[0]) and p[1].requires_grad)
    return params


if __name__ == "__main__":

    def test_data():
        TRAIN_BSON_FILE = './data/train_example.bson'

        # Some parameters
        N_TRAIN, BS, N_THREADS= 82, 2, 1

        # Dataset and loader
        train_example_dataframe = get_data_frame(TRAIN_BSON_FILE, N_TRAIN, True)
        train_dataset = CdiscountTrainDataset(TRAIN_BSON_FILE, train_example_dataframe, transf.ToTensor())
        loader = DataLoader(train_dataset, batch_size=BS, num_workers=N_THREADS, shuffle=True)

        # Let's go fetch some data!
        pbar = tqdm(total=len(loader))
        for batch, target in loader:
            pbar.update()
        pbar.close()

    test_data()