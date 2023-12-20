# coding=utf8
import os
import time

import torchvision.models
import torchvision.transforms as transforms
import numpy as np
import torch
from progress.bar import Bar
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, ResNet18_Weights, GoogLeNet
from sklearn.metrics import precision_score, recall_score
from mixmatch_class import F1Score, AverageMeter, label_dataset, unlabeled_dataset, TransformTwice, test_valid_dataset

# from mixmatch_student import num_classes, epochs, n_labeled_per_class, batch_size

criterion = torch.nn.CrossEntropyLoss()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, epochs, lambda_u):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u) ** 2)

        return Lx, Lu, lambda_u * linear_rampup(epoch, epochs)


def create_model(num_classes, ema=False):
    # Move the model to CUDA, if available
    #     global model
    if torch.cuda.is_available():
        # model = WideResNet(num_classes=num_classes)
        # model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
        model = torchvision.models.resnet50(ResNet50_Weights.DEFAULT)
        # model.fc = nn.Linear(2048, 1024)
        model.fc = nn.Linear(2048, num_classes)
        model.to("cuda")

    if ema == True:
        for param in model.parameters():
            param.detach_()
    return model


def linear_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)


def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets


def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]


macro_f1_score = F1Score(average='macro')
micro_f1_score = F1Score(average='micro')


def student_train(epochs, lambda_u, labeled_train_loader, unlabeled_train_loader, optimizer, ema_optimizer, model, num_classes, epoch, train_iteration=1024, T=0.5, alpha=0.75):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_x = AverageMeter()
    losses_u = AverageMeter()
    ws = AverageMeter()
    end = time.time()

    bar = Bar('Training', max=train_iteration)
    labeled_train_iter = iter(labeled_train_loader)
    unlabeled_train_iter = iter(unlabeled_train_loader)

    model.train()
    for batch_idx in range(train_iteration):
        try:
            inputs_x, targets_x = labeled_train_iter.__next__()
        except:
            labeled_train_iter = iter(labeled_train_loader)
            inputs_x, targets_x = labeled_train_iter.__next__()

        try:
            (inputs_u, inputs_u2), _ = unlabeled_train_iter.__next__()
        except:
            unlabeled_train_iter = iter(unlabeled_train_loader)
            (inputs_u, inputs_u2), _ = unlabeled_train_iter.__next__()

        # measure data loading time
        data_time.update(time.time() - end)

        batch_size = inputs_x.size(0)

        # Transform label to one-hot
        targets_x = torch.zeros(batch_size, num_classes).scatter_(1, targets_x.view(-1, 1).long(), 1).to(device)

        inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda(non_blocking=True)
        inputs_u = inputs_u.cuda()
        inputs_u2 = inputs_u2.cuda()

        with torch.no_grad():
            # compute guessed labels of unlabel samples
            outputs_u = model(inputs_u)
            outputs_u2 = model(inputs_u2)
            p = (torch.softmax(outputs_u, dim=1) + torch.softmax(outputs_u2, dim=1)) / 2
            pt = p ** (1 / T)
            targets_u = pt / pt.sum(dim=1, keepdim=True)
            targets_u = targets_u.detach()

        # mixup
        all_inputs = torch.cat([inputs_x, inputs_u, inputs_u2], dim=0)
        all_targets = torch.cat([targets_x, targets_u, targets_u], dim=0)

        l = np.random.beta(alpha, alpha)

        l = max(l, 1 - l)

        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]

        mixed_input = l * input_a + (1 - l) * input_b
        mixed_target = l * target_a + (1 - l) * target_b

        # interleave labeled and unlabed samples between batches to get correct batchnorm calculation
        mixed_input = list(torch.split(mixed_input, batch_size))
        mixed_input = interleave(mixed_input, batch_size)

        logits = [model(mixed_input[0])]
        for input in mixed_input[1:]:
            logits.append(model(input))

        # put interleaved samples back
        logits = interleave(logits, batch_size)
        logits_x = logits[0]
        logits_u = torch.cat(logits[1:], dim=0)
        train_criterion = SemiLoss()
        # outputs_x, targets_x, outputs_u, targets_u, epoch, epochs, lambda_u
        Lx, Lu, w = train_criterion(logits_x, mixed_target[:batch_size], logits_u, mixed_target[batch_size:], epoch + batch_idx / train_iteration, epochs, lambda_u)

        loss = Lx + w * Lu
        # print('The loss is ', loss)

        # record loss
        losses.update(loss.item(), inputs_x.size(0))
        losses_x.update(Lx.item(), inputs_x.size(0))
        losses_u.update(Lu.item(), inputs_x.size(0))
        ws.update(w, inputs_x.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema_optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        print(
            '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | Loss_x: {loss_x:.4f} | Loss_u: {loss_u:.4f} | W: {w:.4f}'.format(
                batch=batch_idx + 1,
                size=train_iteration,
                data=data_time.avg,
                bt=batch_time.avg,
                total=bar.elapsed_td,
                eta=bar.eta_td,
                loss=losses.avg,
                loss_x=losses_x.avg,
                loss_u=losses_u.avg,
                w=ws.avg,
            ))

    return (losses.avg, losses_x.avg, losses_u.avg,)


# In[53]:
def valid_best(best_acc, val_loader, model, epoch, mode, test_model_path_root, test_model_path, task_type):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    macro_f1_scores = AverageMeter()
    micro_f1_scores = AverageMeter()
    accuracies = AverageMeter()
    precision_1 = AverageMeter()
    recall_1 = AverageMeter()
    if not os.path.exists(test_model_path_root):
        os.makedirs(test_model_path_root, exist_ok=True)
    if task_type == "test":
        print("test task")
        if os.path.exists(test_model_path):
            resume = torch.load(test_model_path)
            model.load_state_dict(resume['state_dict'], strict=False)
            print('Load checkpoint {}'.format(test_model_path))
    model.eval()
    bar = Bar(f'{mode}', max=len(val_loader))
    end = time.time()
    print(f'\n************{mode}*************')
    # best_acc = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)

            # compute output
            outputs = model(inputs)

            # compute loss
            loss = criterion(outputs, targets)

            ## compute metrics
            # accuracy
            probs = torch.softmax(outputs, dim=1)
            predicted_labels = torch.argmax(probs, dim=1)

            correct = predicted_labels == targets
            accuracy = correct.sum() / float(targets.size(0))

            # macro F1 score
            macro_f1 = macro_f1_score(predicted_labels.flatten(), targets.flatten())

            # micro F1 score
            micro_f1 = micro_f1_score(predicted_labels.flatten(), targets.flatten())

            # precision_per_class = precision_score(targets.flatten(),predicted_labels.flatten(), average=None)
            overall_precision = precision_score(targets.cpu().flatten(), predicted_labels.cpu().flatten(), average='micro')
            # recall_per_class = recall_score(targets.flatten(),predicted_labels.flatten(), average=None)
            overall_recall = recall_score(targets.cpu().flatten(), predicted_labels.cpu().flatten(), average='micro')

            # update the loss and metrics
            losses.update(loss.item(), inputs.size(0))
            accuracies.update(accuracy, inputs.size(0))
            precision_1.update(overall_precision, inputs.size(0))
            recall_1.update(overall_recall, inputs.size(0))
            macro_f1_scores.update(macro_f1.item(), inputs.size(0))
            micro_f1_scores.update(micro_f1.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)

            # print the logs
            # if (epoch % 50 == 0 and epoch != 0) or epoch + 1 == len(val_loader):
            print(
                '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | Accuracy: {accuracy:.4f} | Macro_F1_score: {macro_f1_score:.4f} | Micro_F1_score: {micro_f1_score:.4f}'.format(
                    batch=batch_idx + 1,
                    size=len(val_loader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    accuracy=accuracies.avg,
                    macro_f1_score=macro_f1_scores.avg,
                    micro_f1_score=micro_f1_scores.avg
                ))
    print(accuracies.avg, best_acc)
    if (accuracies.avg >= best_acc) & (task_type == "valid"):
        best_acc = accuracies.avg
        save_file_path = test_model_path
        states = {'state_dict': model.state_dict(),
                  'epoch': epoch,
                  'acc': best_acc}
        torch.save(states, save_file_path)
        print('Saved!')
    return best_acc, losses.avg, accuracies.avg, macro_f1_scores.avg, micro_f1_scores.avg, overall_precision, overall_recall


def valid(val_loader, model, epoch, mode):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    macro_f1_scores = AverageMeter()
    micro_f1_scores = AverageMeter()
    accuracies = AverageMeter()
    precision_1 = AverageMeter()
    recall_1 = AverageMeter()

    model.eval()
    bar = Bar(f'{mode}', max=len(val_loader))
    end = time.time()
    print(f'\n************{mode}*************')
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)

            # compute output
            outputs = model(inputs)

            # compute loss
            loss = criterion(outputs, targets)

            ## compute metrics
            # accuracy
            probs = torch.softmax(outputs, dim=1)
            predicted_labels = torch.argmax(probs, dim=1)

            correct = predicted_labels == targets
            accuracy = correct.sum() / float(targets.size(0))

            # macro F1 score
            macro_f1 = macro_f1_score(predicted_labels.flatten(), targets.flatten())

            # micro F1 score
            micro_f1 = micro_f1_score(predicted_labels.flatten(), targets.flatten())

            # precision_per_class = precision_score(targets.flatten(),predicted_labels.flatten(), average=None)
            overall_precision = precision_score(targets.cpu().flatten(), predicted_labels.cpu().flatten(), average='micro')
            # recall_per_class = recall_score(targets.flatten(),predicted_labels.flatten(), average=None)
            overall_recall = recall_score(targets.cpu().flatten(), predicted_labels.cpu().flatten(), average='micro')

            # update the loss and metrics
            losses.update(loss.item(), inputs.size(0))
            accuracies.update(accuracy, inputs.size(0))
            precision_1.update(overall_precision, inputs.size(0))
            recall_1.update(overall_recall, inputs.size(0))
            macro_f1_scores.update(macro_f1.item(), inputs.size(0))
            micro_f1_scores.update(micro_f1.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            if (epoch % 50 == 0 and epoch != 0) or epoch + 1 == len(val_loader):
                # print the logs
                print(
                    '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | Accuracy: {accuracy:.4f} | Macro_F1_score: {macro_f1_score:.4f} | Micro_F1_score: {micro_f1_score:.4f}'.format(
                        batch=batch_idx + 1,
                        size=len(val_loader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        loss=losses.avg,
                        accuracy=accuracies.avg,
                        macro_f1_score=macro_f1_scores.avg,
                        micro_f1_score=micro_f1_scores.avg
                    ))

    return losses.avg, accuracies.avg, macro_f1_scores.avg, micro_f1_scores.avg, overall_precision, overall_recall


def student_load_datasets(label_features,
                          label_targets, unlabeled_features,
                          unlabeled_targets, img_size):
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

    label_datasets = label_dataset(label_features,
                                   label_targets,
                                   transform_s1=train_transform,
                                   target_transform=lambda t: torch.tensor((float(t))).long()
                                   )
    unlabeled_datasets = unlabeled_dataset(unlabeled_features,
                                           unlabeled_targets,
                                           transform_s1=TransformTwice(train_transform),
                                           target_transform=lambda t: torch.tensor((float(t))).long())

    # print(train_datasets.targets)
    return label_datasets, unlabeled_datasets


def labels_unlabeled_split(labels, n_labeled_per_class, num_classes, seed):
    """
    数据划分:划分有标签和无标签
    有标签数据前10个，无标签数据从第21个到倒数20，验证集数据倒数20个
    """
    labels = np.array(labels)
    train_labeled_idx = []
    train_unlabeled_idx = []
    for i in range(num_classes):
        idx = np.where(labels == i)[0]
        # 打乱后取出n个带标签数据
        np.random.shuffle(idx)
        train_labeled_idx.extend(idx[:n_labeled_per_class])
        # 从第n个到最后一个是无标签数据
        train_unlabeled_idx.extend(idx[n_labeled_per_class:])
    # 全部取出后再次打乱（设置seed）
    np.random.seed(seed)
    np.random.shuffle(train_labeled_idx)
    np.random.shuffle(train_unlabeled_idx)

    return train_labeled_idx, train_unlabeled_idx


# Train Dataloaders
def train_data_handle(train_data, unlabeled_train_data, test_data, valid_data, test_transform, img_size, n_labeled_per_class, num_classes, batch_size, seed, n_shot):
    # n_labeled_per_class = n_shot
    # 拆包
    x_train, y_train = train_data
    x_test, y_test = test_data
    x_valid, y_valid = valid_data
    unlabeled_x_train, unlabeled_y_train = unlabeled_train_data
    # 分为有标签数据、无标签数据、验证集
    train_labeled_idx, train_unlabeled_idx = labels_unlabeled_split(y_train, n_labeled_per_class, num_classes, seed)
    print(len(train_labeled_idx), len(train_unlabeled_idx))
    # # # 根据index取出具体的数据和标签
    label_features = x_train[train_labeled_idx]
    label_targets = y_train[train_labeled_idx]
    unlabeled_features = x_train[train_unlabeled_idx]
    unlabeled_targets = y_train[train_unlabeled_idx]
    # -----
    # 分离标签数据
    x_train = label_features
    y_train = label_targets
    # 合并unlabeled数据
    unlabeled_x_train = np.concatenate((unlabeled_x_train, unlabeled_features), axis=0)
    unlabeled_y_train = np.concatenate((unlabeled_y_train, unlabeled_targets), axis=0)
    # -----
    # Train Dataset
    print("x_train:", len(x_train), "unlabeled_x_train:", len(unlabeled_x_train))
    label_datasets, unlabeled_datasets = student_load_datasets(x_train, y_train, unlabeled_x_train, unlabeled_y_train, img_size)
    # label_datasets, unlabeled_datasets = student_load_datasets(label_features, label_targets, unlabeled_features, unlabeled_targets, img_size)
    # # Validation and Test Datasets
    # features: np.array,
    # targets: Union[np.array, None] = None,
    # transform_s1 = None,
    # target_transform = None,
    val_datasets = test_valid_dataset(x_valid, y_valid, data_transform=test_transform, target_transform=lambda t: torch.tensor((float(t))).long())
    test_datasets = test_valid_dataset(x_test, y_test, data_transform=test_transform, target_transform=lambda t: torch.tensor((float(t))).long())

    # val_dataset = labeled_dataset(str(data_path), val_idxs, transform=test_transform, target_transform=lambda t: torch.tensor((float(t))).long())
    # test_dataset = labeled_dataset(str(data_path), transform=test_transform, target_transform=lambda t: torch.tensor((float(t))).long())
    # print(len(test_dataset))
    # print(len(train_labeled_idxs))
    print(len(label_datasets))
    print(len(unlabeled_datasets))
    # print(n_labeled_per_class * num_classes)
    # assert len(label_datasets) == n_labeled_per_class * num_classes
    # print(len(train_labeled_idx))
    # print(len(train_unlabeled_idx))
    # Class labels
    # print(unlabeled_datasets.class_to_idx)

    labeled_train_loader = DataLoader(label_datasets, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)
    unlabeled_train_loader = DataLoader(unlabeled_datasets, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)
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
    # Validation dataloader
    # val_datasets = torchvision.datasets.ImageFolder(root=r"G:\RS1130\data\EuroSAT1\valid-160",transform=train_transform)
    print(len(val_datasets))
    # val_loader = DataLoader(val_datasets, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_datasets, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)

    # Test loader on entire set of n images
    test_loader = DataLoader(test_datasets, batch_size, shuffle=True, pin_memory=True, drop_last=True)

    return labeled_train_loader, unlabeled_train_loader, val_loader, test_loader
