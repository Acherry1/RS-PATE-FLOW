#coding=utf8
import torchvision
from matplotlib.transforms import Transform
import numpy as np
from pathlib import Path
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
from typing import Tuple, Union
from torch.utils.data import Dataset



def load_split(
        data_subdir: Path,
        split: Union[bool, str],
        dataset_class,
):
    # dataset = Datasets(data_subdir)
    if type(split) == bool:
        dataset = dataset_class(root=data_subdir, train=split, download=True)
    elif type(split) == str:
        dataset = dataset_class(root=data_subdir, split=split, download=True)
    else:
        raise RuntimeError(f"Got unknown split type {split}")

    x = dataset.data
    if dataset_class == datasets.SVHN:
        y = dataset.labels
    else:
        y = dataset.targets

    if isinstance(x, torch.Tensor):
        x = x.numpy()
    if isinstance(y, torch.Tensor):
        y = y.numpy()

    return x, y


def load_RS_split(train_dir):
    # """
    # 分割成训练数据和测试数据，并对应标签
    # """

    # train_dir = r'G:\RS1130\data\NWPU-RESISC45\split_NR45\train'
    # test_dir = r'G:\RS1130\data\NWPU-RESISC45\split_NR45\test'
    transform = transforms.Compose([
        # transforms.Resize(64),
        # transforms.RandomResizedCrop(64, scale=(0.5, 1.0)),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_dataset = torchvision.datasets.ImageFolder(train_dir, transform)
    # test_dataset = torchvision.datasets.ImageFolder(test_dir, test_transform)

    # train_dataset, test_dataset = torch.utils.data.random_split(total_data, [train_size, test_size])
    train_dataloader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True, num_workers=1)
    # test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True, num_workers=0)
    for i, (img, label) in enumerate(train_dataloader):
        train_img = (img.numpy()*255).astype(np.uint8)
        train_label = label.numpy()
    # train_img, train_label = add_dataset(train_dir)

    # if isinstance(train_img, torch.Tensor):
    #     x = train_img.cpu().numpy()
    # y = train_label.cpu().numpy()
    # if isinstance(test_img, torch.Tensor):
    #     test_img = test_img.numpy()
    # y_test = y_test.numpy()

    return train_img, train_label


def load_dataset(
        data_subdir: Path,
        dataset_class: Dataset,
):
    """Loads dataset, and concatenates train and test sets

    Args:
        data_subdir (Path): _description_
        dataset_class (Dataset): _description_

    Returns:
        _type_: _description_
    """
    path_x = os.path.join(data_subdir, 'x.npy')
    path_y = os.path.join(data_subdir, 'y.npy')

    if os.path.isfile(path_x) and os.path.isfile(path_y):
        x = np.load(path_x)
        y = np.load(path_y)
        return x, y
    # 加載mnist數據
    # x_train, y_train = load_split(data_subdir, True, datasets.MNIST)
    # x_test, y_test = load_split(data_subdir, False, datasets.MNIST)
    # rs
    # 加載rs數據
    x_train, y_train, x_test, y_test = load_RS_split(data_subdir)

    x = np.concatenate([x_train, x_test])
    y = np.concatenate([y_train, y_test])
    np.save(path_x, x)
    np.save(path_y, y)
    return x, y


import os


def load_RS_dataset(
        data_subdir, train_dir):
    """Loads dataset, and concatenates train and test sets

    Args:
        data_subdir (Path): _description_
        dataset_class (Dataset): _description_

    Returns:
        _type_: _description_
    """
    path_x = os.path.join(data_subdir, 'x.npy')
    path_y = os.path.join(data_subdir, 'y.npy')
    #
    # if os.path.isfile(path_x) and os.path.isfile(path_y):
    #     x = np.load(path_x)
    #     y = np.load(path_y)
    #     return x, y

    x, y = load_RS_split(train_dir)
    # x = np.concatenate([x_train, x_test])
    # y = np.concatenate([y_train, y_test])
    np.save(path_x, x)
    np.save(path_y, y)
    return x, y


import os


def load_mnist(data_dir: Path, ):
    (x, y) = load_dataset(
        data_subdir=os.path.join(data_dir, 'mnist'),
        dataset_class=datasets.MNIST,
    )
    shape = np.shape(x)[1:]
    classes = 10

    return (x, y), shape, classes


def load_fashion_mnist(data_dir: Path, ):
    (x, y) = load_dataset(
        data_subdir=data_dir / 'fashion_mnist',
        dataset_class=datasets.FashionMNIST,
    )
    shape = np.shape(x)[1:]
    classes = 10

    return (x, y), shape, classes


def load_cifar10(data_dir: Path, ):
    (x, y) = load_dataset(
        data_subdir=data_dir / 'cifar10',
        dataset_class=datasets.CIFAR10,
    )
    shape = np.shape(x)[1:]
    classes = 10

    return (x, y), shape, classes


def load_svhn(data_dir: Path, ):
    data_subdir = data_dir / 'svhn'

    path_x = data_subdir / 'x.npy'
    path_y = data_subdir / 'y.npy'

    if path_x.is_file() and path_y.is_file():
        x = np.load(path_x)
        y = np.load(path_y)

    else:
        x_train, y_train = load_split(data_subdir, 'train', datasets.SVHN)
        x_test, y_test = load_split(data_subdir, 'test', datasets.SVHN)

        x = np.concatenate([x_train, x_test])
        y = np.concatenate([y_train, y_test])

        x = np.moveaxis(x, 1, -1)

        np.save(path_x, x)
        np.save(path_y, y)

    shape = np.shape(x)[1:]
    classes = 10

    return (x, y), shape, classes
