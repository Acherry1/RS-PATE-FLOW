import math
from time import time
import numpy as np
from typing import Tuple

from mixmatch_student import mixmatch_student_main

from data_now.parameters import ExperimentParameters
from data_now.pate import average_dp_budgets


def main(
        teacher_num,
        prms: ExperimentParameters,
        data_name,
        train_data: Tuple[np.array],
        unlabeled_train_data: Tuple[np.array],
        test_data: Tuple[np.array],
        valid_data: Tuple[np.array], image_size,
        n_labels, seed
):
    # n_labels是可以查询标签的数量

    # upack data
    # x_train_unlabeled, y_train_unlabeled = train_data_unlabeled
    # x_valid, y_valid = valid_data
    if data_name == "NWPU-RESISC45":
        n_classes = 45
    else:
        n_classes = 10
        n_shot = int(n_labels / n_classes)
    # build model
    # model = ClassifierWrapper(input_size=np.shape(x_train)[1:],
    #                           architecture=prms.models.architecture,
    #                           dataset=prms.data.data_name,
    #                           # n_classes=45)
    #                           n_classes=n_classes)
    # net=
    # model.build()
    # 使它返回test的准确率等信息
    # train_data, test_data, valid_data, img_size, seed, n_shot
    # test_accuracy, save_model, len(labeled_train_loader), len(unlabeled_train_loader), len(test_loader), len(val_loader)
    # mixmatch训练学生模型
    test_accuracy, model, labeled_train_size, unlabeled_train_size, test_size, val_size, overall_precision, overall_recall = mixmatch_student_main(teacher_num, train_data,
                                                                                                                                                   unlabeled_train_data, test_data,
                                                                                                                                                   valid_data, image_size,
                                                                                                                                                   seed, n_shot)

    # 使用与教师机相同的模型训练学生
    # 1。找到教师机的训练模型
    if ""=="":


        pass
    # 2.进行改造
    # 2.1输入数据改造
    # 2.2输出信息更改
    # 2.3保存模型及数据路径更改
    # 2.4确保不会覆盖之前训练数据
    # train model
    # print("x_train_len",len(x_train))
    # batch_size: int = min(64, int(len(x_train)))
    # if batch_size < 16:
    #     logger.warning(f"Found extremely low batch size of {batch_size}.")

    train_start_time = time()
    # train_loss_curve = model.fit(
    #     data_train=(x_train, y_train),
    #     batch_size=batch_size,
    #     n_epochs=prms.models.student_epochs,
    #     lr=prms.models.lr,
    #     weight_decay=prms.models.weight_decay,
    # )
    # train_time = time() - train_start_time
    train_loss_curve = ""

    # test_accuracy, test_accuracy_by_class = model.accuracy(x_test=x_test,
    #                                                        y_test=y_test)
    # test_precision, test_precision_by_class = model.precision(x_test=x_test,
    #                                                           y_test=y_test)
    # test_recall, test_recall_by_class = model.recall(x_test=x_test,
    #                                                  y_test=y_test)
    # print(test_precision)

    avg_budget = average_dp_budgets(epsilons=prms.pate.budgets,
                                    deltas=[prms.pate.delta] *
                                           len(prms.pate.budgets),
                                    weights=list(prms.pate.epsilon_weights))[0]

    # collect statistics
    # TODO: prms removed, log from caller
    # TODO: 'costs', 'n_limit', 'n_labels', 'limit' removed, log from caller
    statistics = {
        'model_architecture': prms.models.architecture,
        'n_data_train': labeled_train_size,
        'unlabeled_size': unlabeled_train_size,
        'test_size': test_size,
        'valid_size': val_size,
        'avg_budget': round(avg_budget, 3),
        'avg_budget_linear': round(math.exp(avg_budget), 3),
        'test_accuracy': test_accuracy,
        # 'test_accuracy_by_class':
        #     [round(a, 3) for a in test_accuracy_by_class],
        'test_precision': overall_precision,
        # 'test_precision_by_class':
        #     [round(a, 3) for a in test_precision_by_class],
        'test_recall': overall_recall,
        # 'test_recall_by_class': [round(a, 3) for a in test_recall_by_class],
        # 'train_loss_curve': train_loss_curve,
    }
    # for key, value in model.__dict__.items():
    #     if key not in ['instance', 'statistics']:
    #         if type(value) not in [list, dict, tuple]:
    #             statistics[key] = value
    #         else:
    #             statistics[key] = str(value)
    #
    # for key, value in model.statistics.items():
    #     if type(value) not in [list, dict, tuple]:
    #         statistics[key] = value
    #     else:
    #         statistics[key] = str(value)

    return model, statistics
