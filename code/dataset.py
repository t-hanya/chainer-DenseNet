# -*- coding: utf-8 -*-


from chainer.dataset import DatasetMixin
from chainer.datasets import cifar
import numpy as np


# calculated from training set
C10_MEAN = np.array([125.3, 123.0, 113.9], dtype=np.float32)
C10_STD = np.array([63.0, 62.1, 66.7], dtype=np.float32)
C100_MEAN = np.array([129.3, 124.1, 112.4], dtype=np.float32)
C100_STD = np.array([68.2,  65.4,  70.4], dtype=np.float32)


def _normalize_data(img, mean, std):
    """Normalize image value with given mean and standard deviation"""

    img = img * 255.
    img = (img - mean[:, None, None]) / std[:, None, None]

    return img


def _apply_data_augmentation(img, pad=4, crop_size=32):
    """Apply data augmentation to given image"""

    # add padding
    padded = np.pad(img,
                    ((0, 0), (pad, pad), (pad, pad)),
                    mode='constant', constant_values=0.)

    # randomly crop
    h, w = img.shape[1:3]
    top = np.random.randint(0, h + pad * 2 - crop_size + 1)
    left = np.random.randint(0, w + pad * 2 - crop_size + 1)
    img = padded[:, top: top + crop_size, left: left + crop_size]

    # flip horizontal with probability of 0.5
    if np.random.rand() > 0.5:
        img = img[:, :, :: -1]

    return img


class CIFARDataset(DatasetMixin):
    """Dataset implementation for training DenseNet"""

    def __init__(self, cifar_dataset, mean, std, augmentation=True):
        self._dataset = cifar_dataset
        self._mean = mean
        self._std = std
        self._augmentation = augmentation

    def __len__(self):
        return len(self._dataset)

    def get_example(self, i):
        img, label = self._dataset[i]
        img = _normalize_data(img, self._mean, self._std)
        if self._augmentation:
            img = _apply_data_augmentation(img)

        return img, label


def get_C10():
    """CIFAR-10 dataset without augmentation"""
    train, test = cifar.get_cifar10()
    train_dataset = CIFARDataset(train, C10_MEAN, C10_STD, False)
    test_dataset = CIFARDataset(test, C10_MEAN, C10_STD, False)

    return train_dataset, test_dataset


def get_C10_plus():
    """CIFAR-10 dataset with data augmentaion"""
    train, test = cifar.get_cifar10()
    train_dataset = CIFARDataset(train, C10_MEAN, C10_STD, True)
    test_dataset = CIFARDataset(test, C10_MEAN, C10_STD, True)

    return train_dataset, test_dataset


def get_C100():
    """CIFAR-100 dataset without augmentation"""
    train, test = cifar.get_cifar100()
    train_dataset = CIFARDataset(train, C100_MEAN, C100_STD, False)
    test_dataset = CIFARDataset(test, C100_MEAN, C100_STD, False)

    return train_dataset, test_dataset


def get_C100_plus():
    """CIFAR-100 dataset with data augmentation"""

    train, test = cifar.get_cifar100()
    train_dataset = CIFARDataset(train, C100_MEAN, C100_STD, True)
    test_dataset = CIFARDataset(test, C100_MEAN, C100_STD, True)

    return train_dataset, test_dataset
