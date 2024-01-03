# coding=UTF8
import os
import pickle
from typing import Dict
import math
import numpy as np
import sklearn
import torch.cuda
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_now.args import args_parser
from data_now.full_model import FullModel

from data_now.parameters import ExperimentParameters
from data_now.pate import PATE, average_dp_budgets
from data_now.teacher_datasets import teacher_load_datasets
from data_now.val import public_vote

args = args_parser()


def load_net(n_classes, net_path):
    with open(file=f'{net_path}/model.pickle', mode='rb') as f:
        net = pickle.load(f)
    # net = FullModel(arch=args.arch,
    #                 n_classes=n_classes,
    #                 mode=args.mode,
    #                 energy_thr=args.energy_thr).cpu()
    net.load_state_dict(torch.load(f'{net_path}/model.pt'), strict=False)
    print('Load checkpoint {}'.format(net_path))
    if net.cuda():
        net = net.cuda()
    return net


def load_best_net(net, net_path):
    if os.path.exists(net_path):
        resume = torch.load(net_path)
        net.load_state_dict(resume['state_dict'], strict=False)
        print('Load checkpoint {}'.format(net_path))
    return net


def linear_costs(costs):
    res = []
    for cs in costs:
        r = []
        for c in cs:
            try:
                linear = round(math.exp(c), 3)
            except OverflowError:
                linear = float('inf')
            r.append(linear)
        res.append(r)
    return res


def main(
        epochs,
        prms: ExperimentParameters,
        num_classes,
        data_name,
        aggregator: str,
        alphas: np.ndarray,
        public_data: np.array,
        budgets_per_sample: Dict,  # TODO: Is this a dict?
        mapping_t2p: Dict,  # TODO: Is this a dict?
        predictions
):
    # unpack public data
    if predictions is not None:
        predictions = predictions
    else:
        x_public_data, y_public_data = public_data
        if data_name == "NWPU-RESISC45":
            n_classes = 45
        else:
            n_classes = 10
        # calculate predictions
        predictions = []
        # model = ClassifierWrapper(input_size=np.shape(x_public_data)[1:],
        #                           architecture=prms.models.architecture,
        #                           dataset=prms.data.data_name,
        #                           n_classes=n_classes,
        #                           seed=prms.pate.seed)
        train_features = None
        train_targets = None
        valid_data = None
        # 1.将数据处理成dataloader（使用teacher数据处理方式）
        _, void_datasets, _ = teacher_load_datasets(train_features,
                                                    train_targets, public_data,
                                                    valid_data,
                                                    args.batch_size,
                                                    args.img_size,
                                                    args.n_workers)
        batch_size = len(public_data[1])
        print("val_batch_size:", batch_size)
        val_loader = DataLoader(dataset=void_datasets, batch_size=batch_size, shuffle=False,
                                num_workers=args.n_workers,
                                drop_last=False)
        net = FullModel(arch=args.arch,
                        n_classes=n_classes,
                        mode="s2",
                        energy_thr=args.energy_thr).cuda()
        best_acc = 0
        for t in tqdm(range(prms.pate.n_teachers),
                      desc='calculate teacher predictions'):
            # for t in range(1):
            # model = model.load(f'{prms.teachers_dir}/teacher_{t}')
            # net = load_net(n_classes, f'{prms.teachers_dir}/teacher_{t}')

            try:
                net_path = args.resume_path.replace('dataset', args.dataset).replace('arch', args.arch).replace('epochs', str(epochs)).replace('teacher-num',
                                                                                                                                               str(prms.pate.n_teachers)).replace(
                    'teacher-id', str(t))
                # net_path = r"G:\RS1130\codeNew\individualized-pate-SKAL\PATE-structure\checkpoints\EuroSAT_googlenet_50_100\EuroSAT_googlenet_50_100_{}.pth".format(t)
                net = load_best_net(net, net_path)
            except:
                # model = model.load(f'{prms.teachers_dir}/teacher_{t}')
                net = load_net(n_classes, f'{prms.teachers_dir}/teacher_{t}')

            # print(len(x_public_data))
            # for i in x_public_data

            criterion = nn.CrossEntropyLoss().cuda()
            # resume_path = args.resume_path.replace('dataset', args.dataset) \
            #     .replace('arch', args.arch)

            # 2.将数据放入validation评估器，返回类别标签
            acc, class_acc, out_put = public_vote(t, val_loader, net, criterion, num_classes)

            # 3.将标签收集
            # for i in out_put:
            predictions.append(out_put.numpy())
        predictions = np.array(predictions)
    # predictions把投票结果保存起来
    # run pate algorithm to vote for labels
    pate = PATE(
        seed=prms.pate.seed,
        n_teachers=prms.pate.n_teachers,
        n_classes=n_classes,
        predictions=predictions,
        budgets=budgets_per_sample,
        mapping=mapping_t2p,
        aggregator_name=aggregator,
        collector_name=prms.pate.collector,
        delta=prms.pate.delta,
        alphas=alphas,
        sigma=prms.pate.sigma,
        n_labels=prms.pate.n_labels,
        sigma1=prms.pate.sigma1,
        t=prms.pate.t,
    )
    pate.prepare()
    pate.predict_all()
    pate.simplify_budgets_costs()

    # label predictions with rejected votes being -1
    y_pred = np.array(pate.labels, dtype=np.longlong)
    # epsilon限制内可以查询的标签的数量
    n_votes = len(y_pred)

    filter_responds = y_pred != -1
    y_pred_clean = y_pred[filter_responds]
    # 选取了从0到可查询的标签数量
    print("n_votes:", n_votes)
    print("filter_responds", filter_responds)
    true_labels = y_public_data[:n_votes][filter_responds]
    label_accuracy = sklearn.metrics.accuracy_score(y_true=true_labels,
                                                    y_pred=y_pred_clean)

    # features = pate.X[:n_votes]
    features = x_public_data[:n_votes]
    unlabeled_features = x_public_data[n_votes:-1]
    unlabeled_targets = [-1 * len(x_public_data[n_votes:-1])]

    avg_budget = average_dp_budgets(epsilons=prms.pate.budgets,
                                    deltas=[prms.pate.delta] *
                                           len(prms.pate.budgets),
                                    weights=list(prms.pate.epsilon_weights))[0]

    # collect results
    # TODO: prms removed, log from caller
    # TODO: 'budgets_linear' removed, is it used at all?
    statistics = {
        'accuracy': label_accuracy,
        'n_votings': n_votes,
        'n_labels': len(y_pred_clean),
        'alpha_curve': str(pate.alpha_history),
        'ratios': str(pate.ratios),
        'avg_budget': round(avg_budget, 3),
        'avg_budget_linear': round(math.exp(avg_budget), 3),
        # TODO: how does 'epsilons' differ from prms.budgets?
        'simple_budgets': str([round(b, 3) for b in pate.simple_budgets]),
        'costs_curve': pate.simple_costs,
        'costs_curve_linear': str(linear_costs(pate.simple_costs)),
    }

    return features, y_pred, statistics, unlabeled_features, unlabeled_targets, predictions
