#!/usr/bin/env python
# coding: utf-8
from torch.optim.lr_scheduler import ReduceLROnPlateau

# <a href="https://colab.research.google.com/github/vasudev-sharma/Expand_AI-Assignment/blob/master/Expand_ai_problem_2.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>


# Mount the drive on Google Colab
# 训练学生机模型

# # install weights and biases, and torchkeras library
# !pip install wandb
# !pip install torchkeras
# !pip install gdown


import wandb
import torch
from torchvision import transforms
import numpy as np
from mixmatch_class import WeightEMA
from PIL import Image

from mixmatch_train_functions import create_model, valid, train_data_handle, student_train

# set seed to reproduce results
# np.random.seed(9)
# @title
# initialize wandb for logging
# get_ipython().system('wandb login e61e9565bc9a2970c0b0bb55976865b93d3e0f9a')

# 每类的查询量
n_labeled_per_class = 5
image_size = 64
batch_size = 64
lr = 0.0001  # 5e-4  /  5e-5
epochs = 30  # 30
log_freq = 10
ema_decay = 0.99
train_iteration = 512  # No of iterations per epoch  512
lambda_u = 75
T = 0.5
num_classes = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # transforms
# train_transform = transforms.Compose(
#     [
#         transforms.Resize(image_size),
#         transforms.RandomCrop(image_size),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ]
# )
test_transform = transforms.Compose(
    [
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
)

# ## Plotting the images


# %matplotlib inline

# # plot the images
# import matplotlib.pyplot as plt
# plt.figure(figsize=(9, 9))
# # plot 9 images
# for i in range(9):
#   image, label = train_labeled_dataset[i]
#   img = image.permute(1, 2, 0)
#   ax = plt.subplot(3, 3, i + 1)
#   ax.imshow(img.numpy())
#   ax.set_title('Class = %s' % CLASSES[int(label.item())])
#   ax.set_xticks([])
#   ax.set_yticks([])
# import os
# os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"
# plt.show()

# @title

# Training


# Set Optimzers, losses, and metrics(macro_F1, micro_F1)
# Define optimizer, loss, macro and micro F1 scores
model = create_model(num_classes)
ema_model = create_model(num_classes, ema=True)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)
criterion = torch.nn.CrossEntropyLoss()
ema_optimizer = WeightEMA(model, ema_model, lr, alpha=ema_decay)


def train_model(model, labeled_train_loader, unlabeled_train_loader, test_loader, val_loader, epochs, log_freq, name):
    print('********Training has started***************')

    wandb.watch(model, log='all')
    # print(model)
    step = 0
    for epoch in range(1, epochs + 1):
        print('\n Epoch: [%d | %d]' % (epoch, epochs))
        # labeled_trainloader, unlabeled_trainloader, optimizer, ema_optimizer, model, num_classes, epoch, train_iteration=1024, T=0.5, alpha=0.75
        train_loss, train_loss_x, train_loss_u = student_train(epochs, lambda_u, labeled_train_loader, unlabeled_train_loader, optimizer, ema_optimizer, model, num_classes,
                                                               epoch=epoch,
                                                               train_iteration=train_iteration)

        _, train_accuracy, train_macro_f1, train_micro_f1, _, _ = valid(labeled_train_loader, ema_model, epoch, mode='Train_stats')

        val_loss, val_accuracy, val_macro_f1, val_micro_f1, _, _ = valid(val_loader, ema_model, epoch, mode='Validation Stats')
        # scheduler.step(val_loss)
        step = train_iteration * (epoch + 1)

        wandb.log({
            'epoch': epoch,

            # Train metrics
            'train_loss': train_loss,
            'train_loss_x': train_loss_x,
            'train_loss_u': train_loss_u,
            'train_accuracy': train_accuracy,
            'train_macro_f1': train_macro_f1,
            'train_micro_f1': train_micro_f1,

            # Validation metrics
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
            'val_macro_f1': val_macro_f1,
            'val_micro_f1': val_micro_f1,

        })
    #     losses.avg, accuracies.avg, macro_f1_scores.avg, micro_f1_scores.avg
    test_loss, test_accuracy, test_macro_f1, test_micro_f1, overall_precision, overall_recall = valid(test_loader, ema_model, epoch=0, mode='Test Stats ')
    print(test_accuracy, overall_precision, overall_recall)
    print('**************Training has Finished**********************')

    # saving the model
    torch.save(model.state_dict(), '{}.h5'.format(name))
    return test_accuracy, overall_precision, overall_recall, model


#   wandb.save('model.h5')

def mixmatch_student_main(teacher_num, train_data, unlabeled_train_data, test_data, valid_data, img_size, seed, n_shot):
    # wandb initialize a new run
    wandb_name = "{}_{}_{}".format(str(teacher_num), len(train_data), seed)
    wandb.init(project='RS-MIXMATCH', entity="menghyin", name=wandb_name)
    wandb.watch_called = False

    config = wandb.config
    config.batch_size = batch_size
    config.epochs = epochs
    config.lr = lr
    config.seed = 42
    config.classes = num_classes
    config.device = device
    config.ema_decay = ema_decay
    config.train_iteration = train_iteration
    config.lambda_u = lambda_u
    config.T = T

    # set seed
    torch.manual_seed(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    print(len(train_data[1]))
    print(len(test_data[1]))
    print(len(valid_data[1]))

    # ----------------------------------
    labeled_train_loader, unlabeled_train_loader, val_loader, test_loader = train_data_handle(train_data, unlabeled_train_data, test_data, valid_data, test_transform, img_size,
                                                                                              n_labeled_per_class,
                                                                                              num_classes, batch_size, seed, n_shot)
    #     处理标签（有标签的带上噪声标签，无标签数据将标签更换为-1，验证数据替换为真实的标签）
    # sanity check labeled train dataloader
    for batch in labeled_train_loader:
        img, target = batch
        print(img.shape)
        print(target.shape)
        print(target.dtype)
        break

    # sanity check unlabeled train dataloader
    for batch in unlabeled_train_loader:
        # print(batch)
        (img1, img2), _ = batch
        print(img1.shape)
        print(img2.shape)
        break
    # ----------------------------------
    test_accuracy, overall_precision, overall_recall, save_model = train_model(model, labeled_train_loader, unlabeled_train_loader, test_loader, val_loader, epochs, log_freq,
                                                                               wandb_name)
    return test_accuracy, save_model, len(labeled_train_loader), len(unlabeled_train_loader), len(test_loader), len(val_loader),overall_precision, overall_recall


if __name__ == '__main__':
    train_data = ""
    test_data = ""
    valid_data = ""
    img_size = ""
    mixmatch_student_main(train_data, test_data, valid_data, img_size)
    wandb.finish()
