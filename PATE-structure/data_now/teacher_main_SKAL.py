# coding=UTF8
# import torch
# import torch.nn as nn
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from data_now.args import args_parser
from data_now.full_model import FullModel
from data_now.train import train
# from args import args_parser
# from full_model import FullModel
# from train import train
from data_now.val import validation_test
from data_now.teacher_datasets import teacher_load_datasets

from collections import OrderedDict
# from models.full_model import *
# from models import full_model
import pdb
from time import time

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

args = args_parser()


def mode_1_2(epochs, n_classes, resume_path, resume_path_root, train_loader, val_loader, mode='s1'):
    net = FullModel(arch=args.arch,
                    n_classes=n_classes,
                    mode=mode,
                    energy_thr=args.energy_thr).cuda()
    # net = FullModel(arch=args.arch,
    #                 n_classes=n_classes,
    #                 mode=args.mode,
    #                 energy_thr=args.energy_thr).cpu()
    # print(os.path.join(resume_path.split("/")[0:-2], "/"))
    if not os.path.exists(resume_path_root):
        os.makedirs(resume_path_root, exist_ok=True)
    if os.path.exists(resume_path):
        resume = torch.load(resume_path)
        net.load_state_dict(resume['state_dict'], strict=False)
        print('Load checkpoint {}'.format(resume_path))

    # criterion and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    # criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(net.get_parameters(), lr=args.lr)
    sche = torch.optim.lr_scheduler.StepLR(optim, step_size=args.step_size)
    best_acc = 0
    all_time = 0
    # best_acc = 0
    # best_acc, val_acc = validation(0, best_acc, val_loader, net, resume_path, criterion)
    # pdb.set_trace()
    # file_name = '{}_{}.txt'.format(args.dataset, args.mode)

    best_acc, val_accuracy, val_accuracy_by_class, out_put, _, _, _, _ = validation_test(0, best_acc, val_loader, net, resume_path, criterion, n_classes, run_type="Validation")
    train_loss_curve, val_loss_curve = [], []
    for i in range(args.start_epoch, epochs):
        beg_time = time()
        train_acc, train_losses = train(i, train_loader, net, optim, criterion, n_classes)
        best_acc, val_accuracy, val_accuracy_by_class, out_put = validation_test(i, best_acc, val_loader, net, resume_path, criterion, n_classes, run_type="Validation")
        end_time = time()
        all_time = all_time + (end_time - beg_time)
        print('training_time: ', all_time)
        train_loss_curve.append(train_losses)
        #

        sche.step()
    return net, train_loss_curve
    # val_epoch = 1


def teacher_main(epochs, teacher_num, teacher_id, train_data, test_data, valid_data, seed):
    # load datasets feat
    # train_list = args.train_list.replace('dataset', args.dataset)
    # val_list = args.val_list.replace('dataset', args.dataset)
    # # 加载数据
    criterion = nn.CrossEntropyLoss().cuda()

    train_features, train_targets = train_data

    torch.manual_seed(seed)
    # train_features, train_targets, test_features, test_targets, seed
    # x_train, y_train = data_train
    # 打乱数据顺序
    np.random.seed(seed)
    p = np.random.permutation(np.arange(len(train_targets)))
    train_features = train_features[p]
    train_targets = train_targets[p]

    train_datasets, test_datasets, val_datasets = teacher_load_datasets(train_features,
                                                                        train_targets, test_data,
                                                                        valid_data,
                                                                        args.batch_size,
                                                                        args.img_size,
                                                                        args.n_workers)

    train_loader = DataLoader(
        dataset=train_datasets,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.n_workers,
        drop_last=False)

    test_loader = DataLoader(
        dataset=test_datasets,
        batch_size=len(test_data[1]),
        shuffle=False,
        num_workers=args.n_workers,
        drop_last=False)
    val_loader = DataLoader(
        dataset=val_datasets,
        batch_size=len(valid_data[1]),
        shuffle=False,
        num_workers=args.n_workers,
        drop_last=False)
    # bulid model
    resume_path_root = args.resume_path_root.replace('dataset', args.dataset).replace('arch', args.arch).replace('epochs', str(epochs)).replace('teacher-num',
                                                                                                                                                str(teacher_num)).replace(
        'teacher-id', str(teacher_id))
    resume_path = args.resume_path.replace('dataset', args.dataset).replace('arch', args.arch).replace('epochs', str(epochs)).replace('teacher-num', str(teacher_num)).replace(
        'teacher-id', str(teacher_id))
    if args.dataset == 'AID':
        n_classes = 30
    elif args.dataset == 'UCM':
        n_classes = 21
    elif args.dataset == 'NWPU-RESISC45':
        n_classes = 45
    elif args.dataset == 'RSSCN7':
        n_classes = 7
    elif args.dataset == 'EuroSAT':
        n_classes = 10
    else:
        n_classes = 0
    train_start_time = time()
    train_loss_total = []
    _, train_loss_curve1 = mode_1_2(epochs, n_classes, resume_path, resume_path_root, train_loader, val_loader, mode="s1")

    net, train_loss_curve2 = mode_1_2(epochs, n_classes, resume_path, resume_path_root, train_loader, val_loader, mode="s2")
    train_loss_total.append(train_loss_curve1)
    train_loss_total.append(train_loss_curve2)
    best_acc = 0
    best_acc, test_accuracy, test_accuracy_by_class, out_put, precision_per_class, overall_precision, recall_per_class, overall_recall = validation_test(0, best_acc, test_loader,
                                                                                                                                                         net, resume_path,
                                                                                                                                                         criterion,
                                                                                                                                                         n_classes, run_type="test")

    print(test_accuracy, test_accuracy_by_class)

    train_time = time() - train_start_time
    statistics = {
        # 'model': model_prms.architecture,
        'train_time': train_time,
        'test_accuracy': round(test_accuracy, 3),
        'test_accuracy_by_class':
            [round(a, 3) for a in test_accuracy_by_class],
        'test_precision': overall_precision,
        'test_precision_by_class': precision_per_class,
        'test_recall': overall_recall,
        'test_recall_by_class': recall_per_class,
        'train_loss_curve': train_loss_total,
    }
    return net, statistics


if __name__ == '__main__':
    train_features = ""
    train_targets = ""
    test_features = ""
    test_targets = ""
    seed = 9
    teacher_main(train_features, train_targets, test_features, test_targets, seed)
