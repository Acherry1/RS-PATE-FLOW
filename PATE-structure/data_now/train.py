# coding=utf8
import torch
from torch.autograd import Variable
from data_now.utils import *
import csv
import os
from pycm import *


class ClassLossMeter(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.losses = [0.0] * self.num_classes
        self.counts = [0] * self.num_classes

    def update(self, class_indices, losses, n=1):
        print(class_indices, type(class_indices))
        print(losses, type(losses))
        for class_index, loss in zip(class_indices, losses):
            self.losses[class_index] += loss * n
            self.counts[class_index] += n

    def average_losses(self):
        averages = [loss / max(count, 1) for loss, count in zip(self.losses, self.counts)]
        return averages


def train(epoch, train_loader, net, optim, criterion, num_classes):
    print('train at epoch {}'.format(epoch))

    net.train()
    losses = AverageMeter()
    accuracies = AverageMeter()
    # num_classes = num_classes  # 例如，有5个类别
    class_losses = ClassLossMeter(num_classes)

    lossess = []
    for i, batch_data in enumerate(train_loader):
        imgs_s1, imgs_s2, labels = batch_data
        imgs_s1 = Variable(imgs_s1.cuda())
        imgs_s2 = Variable(imgs_s2.cuda())
        labels = Variable(labels.cuda())
        # imgs_s1 = Variable(imgs_s1.cpu())
        # imgs_s2 = Variable(imgs_s2.cpu())
        # labels = Variable(labels.cpu())
        logits, _, _ = net(imgs_s1, imgs_s2, is_training=True)

        optim.zero_grad()
        loss = criterion(logits, labels.to(dtype=torch.int64))
        loss.backward()
        optim.step()

        acc = accuracy(logits, labels.to(dtype=torch.int64))
        losses.update(loss.item(), logits.size(0))
        accuracies.update(acc, logits.size(0))
        lossess.append(losses.avg)
        # print(labels)
        # class_losses.update(labels.tolist(), list(loss), len(batch_data))

        if (i % 50 == 0 and i != 0) or i + 1 == len(train_loader):
            print('Train:   Epoch[{}]:{}/{}   Loss:{:.4f}   Accu:{:.2f}%'. \
                  format(epoch, i + 1, len(train_loader), float(losses.avg), float(accuracies.avg) * 100))
    # avg_losses = class_losses.average_losses()
    # print(f'Epoch: {epoch}, Average Losses per Class: {avg_losses}')
    # 重置 ClassLossMeter，准备下一个 epoch
    class_losses.reset()

    return accuracies.avg, lossess
