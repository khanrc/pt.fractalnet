""" Dataset class """
import numpy as np
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms


# Cutout code is burrowed from https://github.com/quark0/darts/blob/master/cnn/utils.py
class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask

        return img


def get_dataset(data, path, aug_lv):
    # dataset class
    if data == 'cifar10':
        dset_cls = dset.CIFAR10
        data_shape = (3, 32, 32, 10)
        MEAN = [0.49139968, 0.48215827, 0.44653124]
        STD = [0.24703233, 0.24348505, 0.26158768]
    elif data == 'cifar100':
        dset_cls = dset.CIFAR100
        data_shape = (3, 32, 32, 100)
        MEAN = [0.50707516, 0.48654887, 0.44091784]
        STD = [0.26733429, 0.25643846, 0.27615047]
    else:
        raise ValueError(data)

    transf = []
    if aug_lv >= 1:
        # horizontal mirroring, [-4, 4] translation
        transf += [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip()
        ]

    # data transforms
    trn_transforms = transforms.Compose(transf + [
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])
    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])

    if aug_lv == 2:
        trn_transforms.transforms.append(Cutout(16))

    # get dataset
    train_data = dset_cls(path, train=True, download=True, transform=trn_transforms)
    valid_data = dset_cls(path, train=False, download=True, transform=val_transforms)

    return train_data, valid_data, data_shape
