import os
from time import time
from torch import nn
from torch.utils.data import DataLoader

from data_now.full_model import FullModel
from data_now.teacher_datasets import teacher_load_datasets
from data_now.train import train
from mixmatch_student import args

import torch
from torch.autograd import Variable
from data_now.utils import *
from sklearn.metrics import confusion_matrix, precision_score, recall_score


def student_validation_test(epoch, best_acc, val_loader, net, student_path, criterion, num_classes, run_type):
    # print('val at epoch {}'.format(epoch))
    net.eval()
    losses = AverageMeter()
    accuracies = AverageMeter()
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    out_put = []

    with torch.no_grad():
        for i, batch_data in enumerate(val_loader):
            imgs_s1, imgs_s2, labels = batch_data
            with torch.no_grad():
                img_s1 = Variable(imgs_s1.cuda())
                img_s2 = Variable(imgs_s2.cuda())
                labels = Variable(labels.cuda())
                logits, _, _ = net(img_s1, img_s2, is_training=False)
                _, classes = torch.nn.functional.softmax(logits, dim=-1).topk(1)
                classes = classes.view(-1).long()
                out_put.append(classes)
                loss = criterion(logits, labels.to(dtype=torch.int64))
                acc = accuracy(logits, labels.to(dtype=torch.int64))
                _, predicted = torch.max(logits, 1)
                for j in range(len(imgs_s1)):
                    label = int(labels.tolist()[j])
                    # print(predicted, type(predicted))
                    prediction = int(predicted.tolist()[j])
                    class_correct[label] += (prediction == label)
                    class_total[label] += 1
            # conf_matrix = confusion_matrix(labels.cpu().numpy(), predicted.cpu().numpy())

            precision_per_class = precision_score(labels.cpu().numpy(), predicted.cpu().numpy(), average=None)
            overall_precision = precision_score(labels.cpu().numpy(), predicted.cpu().numpy(), average='micro')
            recall_per_class = recall_score(labels.cpu().numpy(), predicted.cpu().numpy(), average=None)
            overall_recall = recall_score(labels.cpu().numpy(), predicted.cpu().numpy(), average='micro')
            accuracies.update(acc, logits.size(0))
            losses.update(loss.item(), logits.size(0))
            if (i % 50 == 0 and i != 0) or i + 1 == len(val_loader):
                print('{}:   Epoch[{}]:{}/{}    Loss:{:.4f}   Accu:{:.2f}%'. \
                      format(run_type, epoch, i + 1, len(val_loader), float(losses.avg), float(accuracies.avg) * 100))
        class_acc = [100 * class_correct[i] / max(class_total[i], 1) for i in range(num_classes)]
        print(class_acc)
    out_put = torch.stack(out_put).detach().cpu()
    out_put = torch.squeeze(out_put)
    print('best_acc: {:.2f}%'.format(best_acc * 100))
    print('curr_acc: {:.2f}%'.format(accuracies.avg * 100))
    if (accuracies.avg >= best_acc) and run_type == "Validation":
        best_acc = accuracies.avg
        save_file_path = student_path
        states = {'state_dict': net.state_dict(),
                  'acc': best_acc}
        torch.save(states, save_file_path)
        print('Saved!')

    return best_acc, accuracies.avg, class_acc, out_put, precision_per_class.tolist(), overall_precision, recall_per_class.tolist(), overall_recall


def student_mode_1_2(epochs, n_classes, student_path, student_path_root, train_loader, val_loader, mode='s1', best_acc=0):
    net = FullModel(arch=args.arch,
                    n_classes=n_classes,
                    mode=mode,
                    energy_thr=args.energy_thr).cuda()
    if not os.path.exists(student_path_root):
        os.makedirs(student_path_root, exist_ok=True)
    if os.path.exists(student_path):
        resume = torch.load(student_path)
        net.load_state_dict(resume['state_dict'], strict=False)
        print('Load checkpoint {}'.format(student_path))

    # criterion and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    # criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(net.get_parameters(), lr=args.lr)
    sche = torch.optim.lr_scheduler.StepLR(optim, step_size=args.step_size)

    all_time = 0
    best_acc, val_accuracy, val_accuracy_by_class, out_put, _, _, _, _ = student_validation_test(0, best_acc, val_loader, net, resume_path, criterion, n_classes,
                                                                                                 run_type="Validation")
    train_loss_curve, val_loss_curve = [], []
    for i in range(args.start_epoch, epochs):
        beg_time = time()
        train_acc, train_losses = train(i, train_loader, net, optim, criterion, n_classes)
        best_acc, val_accuracy, val_accuracy_by_class, out_put, _, _, _, _ = student_validation_test(i, best_acc, val_loader, net, resume_path, criterion, n_classes,
                                                                                                     run_type="Validation")
        end_time = time()
        all_time = all_time + (end_time - beg_time)
        print('training_time: ', all_time)
        train_loss_curve.append(train_losses)
        sche.step()
    return net, train_loss_curve, best_acc


def students_main(epochs, teacher_num, teacher_id, train_data, test_data, valid_data, seed):
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
                                                                                                                                                str(teacher_num))
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
    best_acc = 0
    _, train_loss_curve1, best_acc = student_mode_1_2(epochs, n_classes, resume_path, resume_path_root, train_loader, val_loader, mode="s1", best_acc=best_acc)

    net, train_loss_curve2, _ = student_mode_1_2(epochs, n_classes, resume_path, resume_path_root, train_loader, val_loader, mode="s2", best_acc=best_acc)
    train_loss_total.append(train_loss_curve1)
    train_loss_total.append(train_loss_curve2)
    best_acc = 0
    best_acc, test_accuracy, test_accuracy_by_class, out_put, precision_per_class, overall_precision, recall_per_class, overall_recall = student_validation_test(0, best_acc,
                                                                                                                                                                 test_loader,
                                                                                                                                                                 net, resume_path,
                                                                                                                                                                 criterion,
                                                                                                                                                                 n_classes,
                                                                                                                                                                 run_type="test")

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
    pass