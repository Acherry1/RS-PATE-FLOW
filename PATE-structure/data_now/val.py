import torch
from torch.autograd import Variable
from data_now.utils import *
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from pycm import *
import numpy as np
import torch.nn.functional as F


def validation_test(epoch, best_acc, val_loader, net, resume_path, criterion, num_classes, run_type):
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
                # loss = criterion(logits, labels.to(dtype=torch.int64))
                acc = accuracy(logits, labels.to(dtype=torch.int64))
                _, predicted = torch.max(logits, 1)
                for j in range(len(imgs_s1)):
                    label = int(labels.tolist()[j])
                    # print(predicted, type(predicted))
                    prediction = int(predicted.tolist()[j])
                    class_correct[label] += (prediction == label)
                    class_total[label] += 1
            conf_matrix = confusion_matrix(labels, predicted)

            precision_per_class = precision_score(labels, predicted, average=None)
            overall_precision = precision_score(labels, predicted, average='micro')
            recall_per_class = recall_score(labels, predicted, average=None)
            overall_recall = recall_score(labels, predicted, average='micro')
            accuracies.update(acc, logits.size(0))

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
        save_file_path = resume_path
        states = {'state_dict': net.state_dict(),
                  'acc': best_acc}
        torch.save(states, save_file_path)
        print('Saved!')

    return best_acc, accuracies.avg, class_acc, out_put, precision_per_class.tolist(), overall_precision, recall_per_class.tolist(), overall_recall


def public_vote(teacher_id, val_loader, net, criterion, num_classes):
    # print('val at epoch {}'.format(epoch))
    net.eval()
    losses = AverageMeter()
    accuracies = AverageMeter()
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    out_put = []

    with torch.no_grad():
        for i, batch_data in enumerate(val_loader):
            # print(imgs_s1, imgs_s2, labels)
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

            losses.update(loss, logits.size(0))
            accuracies.update(acc, logits.size(0))

            if (i % 50 == 0 and i != 0) or i + 1 == len(val_loader):
                print('voting:   teacher[{}]:{}/{}    Loss:{:.4f}   Accu:{:.2f}%'. \
                      format(teacher_id, i + 1, len(val_loader), float(losses.avg), float(accuracies.avg) * 100))

        class_acc = [100 * class_correct[i] / max(class_total[i], 1) for i in range(num_classes)]
        print(class_acc)
    out_put = torch.stack(out_put).detach().cpu()
    out_put = torch.squeeze(out_put)
    # for i,acc in enumerate(class_acc):
    #     print(acc)
    # print('best_acc: {:.2f}%'.format(best_acc * 100))
    print('curr_acc: {:.2f}%'.format(accuracies.avg * 100))
    # if accuracies.avg >= best_acc:
    #     best_acc = accuracies.avg
    #     save_file_path = resume_path
    #     states = {'state_dict': net.state_dict(),
    #               'acc': best_acc}
    #     # torch.save(states, save_file_path)
    #     print('Saved!')

    return accuracies.avg, class_acc, out_put
