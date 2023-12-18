# coding=UTF8
from typing import Union

import numpy as np
import torch
import torch.utils.data as data
import os
import torchvision.transforms as transforms
from PIL import Image
import random

from torchvision.transforms.functional import to_pil_image


class MultiScaleRandomCrop(object):
    def __init__(self, scales, size):
        self.scales = scales
        self.crop_size = size

    def __call__(self, img):
        img_size = img.size[0]
        scale = random.sample(self.scales, 1)[0]
        re_size = int(img_size / scale)
        img = img.resize((re_size, re_size), Image.BILINEAR)
        x1 = random.randint(0, re_size - img_size)
        y1 = random.randint(0, re_size - img_size)
        x2 = x1 + self.crop_size
        y2 = y1 + self.crop_size
        img = img.crop((x1, y1, x2, y2))
        return img


def make_list(root, split_path):
    list_path = os.path.join(root, split_path)
    data_list = []
    class_dict = {}
    f = open(list_path, 'r')
    line = f.readline()
    while line:
        sample = {}
        line = line.strip('\n')
        img_path, label = line.split(' ')

        sample['img_path'] = img_path
        sample['label'] = label
        data_list.append(sample)
        if label not in class_dict.keys():
            class_dict[label] = [img_path]
        else:
            class_dict[label].append(img_path)

        line = f.readline()
    f.close()
    return data_list, class_dict


class datasets(data.Dataset):

    def __init__(self,
                 features: np.array,
                 targets: Union[np.array, None] = None,
                 data_type="train",
                 transform_s1=None,
                 transform_s2=None,
                 target_transform=None,
                 device: str = 'cuda'):
        if targets is not None:
            assert len(features) == len(targets)
        # self.features = features
        # self.targets = targets
        self.features = torch.Tensor(features)
        self.targets = torch.Tensor(targets)
        self.transform1 = transform_s1
        self.transform2 = transform_s2
        self.target_transform = target_transform
        self.device = device
        self.data_type = data_type
        # print(type(self.features[1]))
        # print(features[1].shape())
        # features = features[0].reshape(12288,)
        # features = Image.fromarray(features)
        # features.show()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # sample = {'data': self.features[idx]}
        # sample['lables'] = self.targets[idx]
        features = self.features[idx]
        features = to_pil_image(features)
        if self.data_type == "train":
            features1 = self.transform1(features)
            img_s1 = features1.float()
            features2 = self.transform2(features)
            img_s2 = features2.float()

        elif self.data_type == "test":
            features1 = self.transform1(features)
            img_s1 = features1.float()

            features2 = self.transform2(features)
            img_s2 = features2.float()
        else:
            print("img_s1，img_s2这里有错误！" * 10)
            img_s1 = None
            img_s2 = None
        # 这里有问题self.transform(features)
        # if self.transform:
        #     print(features.shape)
        #     features = self.transform(features)
        #     # print("y" * 10)
        #     features = features.float()

        # if self.targets is None:
        #     return features

        target = self.targets[idx]
        # if self.target_transform:
        #     target = self.target_transform(target)

        return img_s1, img_s2, target


def teacher_load_datasets(train_features,
                          train_targets, test_data,
                          valid_data, batch_size, img_size, n_workers):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    train_transform = transforms.Compose([
        transforms.Resize(int(img_size)),
        transforms.CenterCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    val_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        normalize,
    ])
    transform_s2 = transforms.Compose([
        transforms.Resize(int(img_size * 2)),
        transforms.CenterCrop(int(img_size * 2)),
        transforms.ToTensor(),
        normalize,
    ])

    # test_features, test_targets = test_data
    # valid_features, valid_targets = valid_data
    if (train_features is None) and (valid_data is None):
        test_features, test_targets = test_data
        # val_datasets = datasets(valid_features,
        #                         valid_targets,
        #                         data_type="test",
        #                         transform_s1=val_transform,
        #                         transform_s2=transform_s2)
        test_datasets = datasets(test_features,
                                 test_targets,
                                 data_type="test",
                                 transform_s1=val_transform,
                                 transform_s2=transform_s2)
        train_datasets = None
        val_datasets = None
    elif train_features is None:
        test_features, test_targets = test_data
        valid_features, valid_targets = valid_data
        # train_datasets = datasets(train_features,
        #                           train_targets,
        #                           data_type="train",
        #                           transform_s1=train_transform,
        #                           transform_s2=transform_s2
        #                           )
        test_datasets = datasets(test_features,
                                 test_targets,
                                 data_type="test",
                                 transform_s1=val_transform,
                                 transform_s2=transform_s2)
        val_datasets = datasets(valid_features,
                                valid_targets,
                                data_type="test",
                                transform_s1=val_transform,
                                transform_s2=transform_s2)
        train_datasets = None
    else:
        test_features, test_targets = test_data
        valid_features, valid_targets = valid_data

        train_datasets = datasets(train_features,
                                  train_targets,
                                  data_type="train",
                                  transform_s1=train_transform,
                                  transform_s2=transform_s2
                                  )
        test_datasets = datasets(test_features,
                                 test_targets,
                                 data_type="test",
                                 transform_s1=val_transform,
                                 transform_s2=transform_s2)
        val_datasets = datasets(valid_features,
                                valid_targets,
                                data_type="test",
                                transform_s1=val_transform,
                                transform_s2=transform_s2)
    # print(train_datasets.targets)
    return train_datasets, test_datasets, val_datasets
