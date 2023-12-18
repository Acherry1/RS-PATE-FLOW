# coding=utf8
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from typing import Tuple, Union

from torchvision.transforms.functional import to_pil_image


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0, activate_before_residual=False):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes, momentum=0.001)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes, momentum=0.001)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None
        self.activate_before_residual = activate_before_residual

    def forward(self, x):
        if not self.equalInOut and self.activate_before_residual == True:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0, activate_before_residual=False):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate, activate_before_residual)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate, activate_before_residual):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate, activate_before_residual))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, num_classes, depth=64, widen_factor=2, dropRate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate, activate_before_residual=True)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3], momentum=0.001)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)


# Macro F1 score PyTorch

class F1Score:
    """
    Class for f1 calculation in Pytorch.
    """

    def __init__(self, average: str = 'weighted'):
        """
        Init.

        Args:
            average: averaging method
        """
        self.average = average
        if average not in [None, 'micro', 'macro', 'weighted']:
            raise ValueError('Wrong value of average parameter')

    @staticmethod
    def calc_f1_micro(predictions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Calculate f1 micro.

        Args:
            predictions: tensor with predictions
            labels: tensor with original labels

        Returns:
            f1 score
        """
        true_positive = torch.eq(labels, predictions).sum().float()
        f1_score = torch.div(true_positive, len(labels))
        return f1_score

    @staticmethod
    def calc_f1_count_for_label(predictions: torch.Tensor,
                                labels: torch.Tensor, label_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate f1 and true count for the label

        Args:
            predictions: tensor with predictions
            labels: tensor with original labels
            label_id: id of current label

        Returns:
            f1 score and true count for label
        """
        # label count
        true_count = torch.eq(labels, label_id).sum()

        # true positives: labels equal to prediction and to label_id
        true_positive = torch.logical_and(torch.eq(labels, predictions),
                                          torch.eq(labels, label_id)).sum().float()
        # precision for label
        precision = torch.div(true_positive, torch.eq(predictions, label_id).sum().float())
        # replace nan values with 0
        precision = torch.where(torch.isnan(precision),
                                torch.zeros_like(precision).type_as(true_positive),
                                precision)

        # recall for label
        recall = torch.div(true_positive, true_count)
        # f1
        f1 = 2 * precision * recall / (precision + recall)
        # replace nan values with 0
        f1 = torch.where(torch.isnan(f1), torch.zeros_like(f1).type_as(true_positive), f1)
        return f1, true_count

    def __call__(self, predictions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Calculate f1 score based on averaging method defined in init.

        Args:
            predictions: tensor with predictions
            labels: tensor with original labels

        Returns:
            f1 score
        """

        # simpler calculation for micro
        if self.average == 'micro':
            return self.calc_f1_micro(predictions, labels)

        f1_score = 0
        for label_id in range(1, len(labels.unique()) + 1):
            f1, true_count = self.calc_f1_count_for_label(predictions, labels, label_id)

            if self.average == 'weighted':
                f1_score += f1 * true_count
            elif self.average == 'macro':
                f1_score += f1

        if self.average == 'weighted':
            f1_score = torch.div(f1_score, len(labels))
        elif self.average == 'macro':
            f1_score = torch.div(f1_score, len(labels.unique()))

        return f1_score


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class WeightEMA(object):
    def __init__(self, model, ema_model, lr, alpha=0.999):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())
        self.wd = 0.02 * lr

        for param, ema_param in zip(self.params, self.ema_params):
            param.data.copy_(ema_param.data)

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for param, ema_param in zip(self.params, self.ema_params):
            if ema_param.dtype == torch.float32:
                ema_param.mul_(self.alpha)
                ema_param.add_(param * one_minus_alpha)
                # customized weight decay
                param.mul_(1 - self.wd)


class TransformTwice:

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2


class label_dataset(data.Dataset):

    def __init__(self,
                 features: np.array,
                 targets: Union[np.array, None] = None,
                 transform_s1=None,
                 target_transform=None,
                 device: str = 'cuda'):
        if targets is not None:
            assert len(features) == len(targets)
        # self.features = features
        # self.targets = targets
        self.features = torch.Tensor(features)
        self.targets = torch.Tensor(targets)
        self.transform1 = transform_s1
        self.target_transform = target_transform
        self.device = device

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        features = self.features[idx]
        features = to_pil_image(features)
        features = self.transform1(features)
        img = features.float()
        target = self.targets[idx]
        if self.target_transform:
            target = self.target_transform(target)
        return img, target


class unlabeled_dataset(data.Dataset):

    def __init__(self,
                 features: np.array,
                 targets: Union[np.array, None] = None,
                 transform_s1=None,
                 target_transform=None,
                 device: str = 'cuda'):
        if targets is not None:
            assert len(features) == len(targets)
        # self.features = features
        # self.targets = targets
        self.features = torch.Tensor(features)
        self.targets = torch.Tensor(targets)
        self.transform1 = transform_s1
        self.target_transform = target_transform
        self.device = device

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        images = self.features[idx]
        images = to_pil_image(images)
        image1, image2 = self.transform1(images)

        '''将每一个标签换成-1'''
        img1 = image1.float()
        img2 = image2.float()
        # self.targets[idx] = -1
        labeled = self.targets[idx]
        if self.target_transform:
            labeled = self.target_transform(labeled)
        return (img1, img2), labeled


class test_valid_dataset(data.Dataset):

    def __init__(self,
                 features: np.array,
                 targets: Union[np.array, None] = None,
                 data_transform=None,
                 target_transform=None,
                 device: str = 'cuda'):
        if targets is not None:
            assert len(features) == len(targets)
        # self.features = features
        # self.targets = targets
        self.features = torch.Tensor(features)
        self.targets = torch.Tensor(targets)
        self.data_transform = data_transform
        self.target_transform = target_transform
        self.device = device

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        images = self.features[idx]
        images = to_pil_image(images)
        images = self.data_transform(images)
        img = images.float()
        labels = self.targets[idx]
        if self.target_transform:
            labels = self.target_transform(labels)

        return img, labels

# # 标签数据的加载
# class labeled_dataset(datasets.ImageFolder):
#
#     def __init__(self, root, indexs=None,
#                  transform=None, target_transform=None
#                  , is_valid_file=None):
#         super(labeled_dataset, self).__init__(root,
#                                               transform=transform, target_transform=target_transform,
#                                               is_valid_file=is_valid_file)
#
#         #         if indexs is not None:
#         #             self.samples = np.array(self.imgs)[indexs].tolist()
#         #             # print(len(self.samples))
#         #             self.targets = np.array(self.targets)[indexs]
#         #             # print(type(self.targets[0]))
#         if indexs is not None:
#             # self.samples = np.array(self.imgs)[indexs].tolist()
#             #             print(indexs)
#             samples = np.array(self.imgs)[indexs].tolist()
#             # self.targets = np.array(self.targets)[indexs]
#             agg_label_xlsx = data_path_xlsx
#             self.samples, self.targets = replace_labels(samples, agg_label_xlsx)
#
#     #         else:
#     #             print("test")
#     #             self.samples = np.array(self.imgs)[indexs].tolist()
#     #             self.targets = np.array(self.targets)[indexs]

# def __getitem__(self, index):
#     """
#     Args:
#         index (int): Index
#
#     Returns:
#         tuple: (image, target) where target is index of the target class.
#     """
#     img, target = super(labeled_dataset, self).__getitem__(index)
#
#     return img, target


# # 无标签的数据加载
# class unlabeled_dataset(datasets.ImageFolder):
#
#     def __init__(self, root, indexs=None, transform=None, target_transform=None, is_valid_file=None):
#         super(unlabeled_dataset, self).__init__(root,
#                                                 transform=transform, target_transform=target_transform,
#                                                 is_valid_file=is_valid_file)
#         #         self.targets = np.array([-1 for i in range(len(self.targets))])
#         if indexs is not None:
#             # self.samples = np.array(self.imgs)[indexs].tolist()
#             samples = np.array(self.imgs)[indexs].tolist()
#             # self.targets = np.array(self.targets)[indexs]
#             agg_label_xlsx = data_path_xlsx
#             self.samples, self.targets = replace_labels(samples, agg_label_xlsx)
#         # print(self.samples)
#         self.targets = np.array([-1 for i in range(len(self.targets))])
#         # print(self.targets)
