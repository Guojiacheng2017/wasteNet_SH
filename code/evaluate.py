
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import SubsetRandomSampler
import os.path

import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sn
import itertools # 绘制混淆矩阵
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score,recall_score, f1_score,roc_auc_score

from config_training import config
from model import ConvNet, wasteModel_CNN, ResNet, ResBlock

# dataset path
test_data_path = config['test_data_path']
model_path = config['model_path']
image_size = config['image_size']

# load model path
# model_load_path = ''

batch_size = 32
test_percentage = 0.001

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# device = torch.device("cpu")

# Mean and std
mean_r = 0.659
mean_g = 0.617
mean_b = 0.587
std_r = 0.223
std_g = 0.221
std_b = 0.228

CLASSES = ['dry_trash', 'poison_trash', 'recycle_trash', 'wet_trash']
class_num = len(CLASSES)
cm = torch.zeros(class_num, class_num)
precision = np.zeros(class_num)
recall = np.zeros(class_num)
f1_score = np.zeros(class_num)
FPR = np.zeros(class_num)
auc = np.zeros(50)
accu = np.zeros(50)


'''Define the method of preprocessing'''
# Other processing
transform_random = [
    transforms.RandomHorizontalFlip(),  # Flip horizontal
    transforms.RandomVerticalFlip(),  # Flip vertical
    transforms.RandomRotation(30),  # Rotation randomly in 30°
    transforms.RandomCrop([image_size, image_size]),  # Crop randomly
    transforms.GaussianBlur(kernel_size=5, sigma=(10.0, 10.0)),  # Gaussian blur
    transforms.Grayscale(3)  # To gray
]
transform_random = transforms.RandomChoice(transform_random)
# Conventional processing and one of other processing
transform_normal = [
    transforms.Resize([image_size, image_size]),  # Resetting image resolution
    transforms.ToTensor(),  # To [0, 1]
    transforms.Normalize([mean_r, mean_g, mean_b], [std_r, std_g, std_b]),
    transform_random
]
transform = transforms.Compose(transform_normal)

# load image data
data_all = torchvision.datasets.ImageFolder(root=test_data_path, transform=transform)
#
# def dataset_test(dataset)
#     sample_num = len(dataset)
#     file_idx = list(range(sample_num))


def dataset_sampler(dataset, test_percentage):
    """
    split dataset into train set, test set
    :param dataset
    :return: split sampler
    """
    sample_num = len(dataset)
    file_idx = list(range(sample_num))
#     train_idx, test_idx = train_test_split(file_idx, test_size=test_percentage, random_state=42)
    train_idx, val_idx = train_test_split(file_idx, test_size=test_percentage, random_state=42)
    # all_sampler = SubsetRandomSampler(file_idx)
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(val_idx)
#     test_sampler = SubsetRandomSampler(test_idx)
    print(sample_num)
    print(len(train_sampler))
    print(len(test_sampler))
    return train_sampler, test_sampler


# func that can move data format into list
def data_format(data_loader):
    '''
    Function that can move data format into list
    :param data_loader:
    :return: data_input, data_label
    '''
    data = data_loader
    data_input = []
    data_label = []
    for _, data in enumerate(data):
        data_input.append(data[0].to(device))
        data_label.append(data[1].to(device))

    return data_input, data_label


@torch.no_grad()
def evaluate_model(y_pred, y, k):
    n = y.shape[0]
    # find the prediction class label
    _, pred_class = y_pred.max(dim=1)
    # correct = (pred_class == y).sum().item()

#     np_y = y
#     np_pred_class = pred_class
    
    np_y = y.detach().cpu().numpy()
    np_pred_class = pred_class.detach().cpu().numpy()
    
    correct = 0
    for j in range(len(np_y)):
        if np_pred_class[j] == np_y[j]:
            correct += 1

    accu = round((correct / n), 3)


#     cm = confusion_matrix(np_y, np_pred_class)
    cm = confusion_matrix(np_y, np_pred_class)

    for j in list(range(0, class_num)):
        precision[j] = round((cm[j, j] / cm[j, :].sum()), 3)
        recall[j] = round(cm[j, j] / cm[:, j].sum(), 3)
        f1_score[j] = round((2 * recall[j] * precision[j] / (recall[j] + precision[j])), 3)
        FPR[j] = round((cm[:, j].sum() - cm[j, j]) / (cm.sum() - cm[j, :].sum()), 3)

    print("accu: %.3f" % accu)

    plot_conf_matrix(k, cm, classes=CLASSES,
                     normalize=False, title='Normalized confusion matrix')

    # print("pred_class: " + str(np_pred_class))
    # print(np_y)
    print("precision: " + str(precision))
    print("recall: " + str(recall))
    print("f1_score: " + str(f1_score))
    print("FPR: " + str(FPR))

#     y_one_hot = np.zeros(len(np_y), class_num)
    y_one_hot = label_binarize(np_y, np.arange(class_num))
#     print(y_one_hot.shape, y_pred.detach().cpu().numpy().shape)
    
    auc = round(roc_auc_score(y_one_hot, y_pred.detach().cpu().numpy(), average='micro'),3)

    print("auc: "  + str(auc))

    fpr, tpr, thresholds = roc_curve(y_one_hot.ravel(), y_pred.detach().cpu().numpy().ravel())  # ravel()表示平铺开来
    plt.figure()
    plt.plot(fpr, tpr, linewidth=2, label='AUC=%.3f' % auc)

    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1.1, 0, 1.1])
    plt.xlabel('False Postivie Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.savefig('../out_fig/resNet_ROC_'+str(k)+'.png')
#     plt.show()

    return accu, auc


def plot_conf_matrix(k,cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Input
    - cm : 计算出的混淆矩阵的值
    - classes : 混淆矩阵中每一行每一列对应的列
    - normalize : True:显示百分比, False:显示个数
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.xlabel('True label')
    plt.ylabel('Predicted label')
    plt.savefig('../out_fig/resNet_confusion_matrix_'+str(k)+'.png')
# ——————————————————————————————————————————————————————————————————


# get all the training, validation and test set here
test_sampler, train_sampler = dataset_sampler(data_all, test_percentage)
# loader = torch.utils.data.DataLoader(data_all, batch_size=batch_size, num_workers=0, sampler=all_sampler)
# train_loader = torch.utils.data.DataLoader(data_all, batch_size=batch_size, num_workers=0, sampler=train_sampler)
test_loader = torch.utils.data.DataLoader(data_all, batch_size=batch_size, num_workers=0, sampler=test_sampler)


# print(test_loader)
# model
resNet = ResNet(ResBlock, img_size=image_size).to(device)
wasteCNN = wasteModel_CNN(image_size).to(device)
CNN = ConvNet(image_size).to(device)
net = resNet

with torch.no_grad():
    for k in list(range(1, 51)):
        print(k)
        model_load_path = '../model/res18_epoch/model_epoch_' + str(k)
        # load model weight
        weight = torch.load(model_load_path, map_location=device)
        net.load_state_dict(weight['model'])
        net.eval()

        # batch_num = 0
        test_pred_eval = torch.zeros(1, 4)
        labels_test_eval = torch.zeros(1, 4)

        for i, data in enumerate(test_loader):
            torch.cuda.empty_cache()         #############
            inputs_test = (data[0].to(device))
            labels_test = (data[1].to(device)).long() #.detach().numpy()  # .reshape(inputs.shape[0], -1)
            test_pred = net(inputs_test)
            if i == 0:
                test_pred_eval = test_pred
                labels_test_eval = labels_test

            else:
                test_pred_eval = torch.cat((test_pred_eval, test_pred), 0)
                labels_test_eval = torch.cat((labels_test_eval, labels_test), 0)

#             print(i)

            # batch_num = i
            # labels_test[i] = labels_test
            # test_pred[i] = test_pred.detach().numpy()

            # print(labels_test.shape)
            # print(test_pred.shape)
            # print(labels_test_eval.shape)
#             print(test_pred_eval.shape)
#             print(inputs_test.shape)

        accu[k-1], auc[k-1] = evaluate_model(test_pred_eval, labels_test_eval, k)

plt.figure()
plt.plot(np.arange(len(accu)), accu, 'ro',np.arange(len(accu)), accu)
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('Test Accuracy')
plt.savefig('../out_fig/resNet_TestAccuracy.png')
# plt.show()

plt.figure()
plt.plot(np.arange(len(auc)), auc, 'ro',np.arange(len(auc)), auc)
plt.xlabel('epoch')
plt.ylabel('AUC')
plt.title('Test AUC')
plt.savefig('../out_fig/resNet_AUC.png')
# plt.show()
print('-----------------------------------------------------------------------------------------')

