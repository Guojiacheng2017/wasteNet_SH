# Created by Jiacheng Guo at Dec  5 23:33:39 CST 2021
# Life first, G2 second, Code third, LECLERC forth
import os.path

from model import ConvNet, wasteModel_CNN, ResNet, ResBlock
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import SubsetRandomSampler

import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split

from config_training import config
import pandas as pd


'''Define global variable'''
# dataset path
data_path = config['data_path']
model_path = config['model_path']
# image size
image_size = config['image_size']  # 128

# label

# dry_trash = 1
# poison_trash = 2
# recycle_trash = 3
# wet_trash = 4



# Data split
# test_percentage = config['test_percentage'] # train = 0.8, test = 0.2
val_percentage = config['val_percentage']   # train = 0.8, val = 0.2
batch_size = config['batch_size']

# Use CPU or GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Mean and std
mean_r = 0.659
mean_g = 0.617
mean_b = 0.587
std_r = 0.223
std_g = 0.221
std_b = 0.228

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
data_all = torchvision.datasets.ImageFolder(root=data_path, transform=transform)



def dataset_sampler(dataset, val_percentage): #  test_percentage,
    """
    split dataset into train set, val set, test set
    :param dataset
    :return: split sampler
    """
    sample_num = len(dataset)
    file_idx = list(range(sample_num))
#     train_idx, test_idx = train_test_split(file_idx, test_size=test_percentage, random_state=42)
    train_idx, val_idx = train_test_split(file_idx, test_size=val_percentage, random_state=42)
    # all_sampler = SubsetRandomSampler(file_idx)
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)
#     test_sampler = SubsetRandomSampler(test_idx)
    return train_sampler, val_sampler #, test_sampler



# get all the training, validation and test set here
train_sampler, val_sampler = dataset_sampler(data_all, val_percentage) # , test_sampler
# loader = torch.utils.data.DataLoader(data_all, batch_size=batch_size, num_workers=0, sampler=all_sampler)
train_loader = torch.utils.data.DataLoader(data_all, batch_size=batch_size, num_workers=0, sampler=train_sampler)
val_loader = torch.utils.data.DataLoader(data_all, batch_size=batch_size, num_workers=0, sampler=val_sampler)
# test_loader = torch.utils.data.DataLoader(data_all, sampler=test_sampler)

# train_tensor = torch.tensor(train_loader, dtype=torch.float32)
# val_tensor = torch.tensor(val_loader, dtype=torch.float32)
# test_tensor = torch.tensor(test_loader, dtype=torch.float32)

#--------------------------------------------------------------------------------------------------#

n_epoch = config['epoch']
batch_size = config['batch_size']
# n_iteration = np.floor(len(data_all)*(1 - val_percentage) / batch_size)


resNet = ResNet(ResBlock, img_size=image_size).to(device)
wasteCNN = wasteModel_CNN(image_size).to(device)
CNN = ConvNet().to(device)

net = resNet

if torch.cuda.is_available():
    net = net.cuda()
    net = net.to(device)

print("model structure:", net)

optimizer = optim.Adam(net.parameters(), lr=config['lr'])
criterion = nn.CrossEntropyLoss()


# for i, data in enumerate(train_loader):
#     inputs = data[0].to(device)
#     labels = data[1].to(device)


import time
train_loss = []
val_loss = []
train_loss_list = np.zeros(n_epoch)
train_accu_list = np.zeros(n_epoch)
val_loss_list = np.zeros(n_epoch)
val_accu_list = np.zeros(n_epoch)

# ------------------------------------------------------- #

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

# ------------------------------------------------------- #
# val_input = []
# val_label = []
# for _, data in enumerate(val_loader):
#     val_input.append(data[0].to(device))
#     val_label.append(data[1].to(device))

# ------------------------------------------------------- #
# save the parameter at the ckpt file
def save(epoch, net, path):
    """
        save parameters into files after each epoch
        :param: epoch seq number, net model, model save path
        :return: None
    """
    stats = {
        'epoch': epoch,
        'model': net.state_dict()
    }
    if not os.path.exists(path):
        os.mkdir(path)

    savePath = os.path.join(path, 'model_epoch_{}'.format(epoch + 1))
    torch.save(stats, savePath)
    print("Saving checkpoints in {}".format(savePath))

# ------------------------------------------------------- #
from tqdm import tqdm

print("# ------------------------------------------------------- #")

# epoch
for idx in range(n_epoch):
    print("epoch:", idx)
    start_time = time.time()
    # iteration
    for i, data in tqdm(enumerate(train_loader)):
        inputs_train = (data[0].to(device))
        labels_train = (data[1].to(device)).long() # .reshape(inputs.shape[0], -1)
        train_pred = net(inputs_train)
        # print(inputs.size())
        # print(i, type(inputs_train))
        # print(train_pred, labels)
        # print(train_pred.size(), labels.size())
        train_loss_i = criterion(train_pred, labels_train)
#         print(train_loss_i)

        train_loss.append(train_loss_i)

        optimizer.zero_grad()
        train_loss_i.backward()
        optimizer.step()

    # compute average training loss
    ave_train_loss = np.sum(train_loss) / len(train_loss)
    train_loss_list[idx] = ave_train_loss

    for i, data in enumerate(val_loader):
        inputs_val = (data[0].to(device))
        labels_val = (data[1].to(device)).long() # .reshape(inputs.shape[0], -1)
        val_pred = net(inputs_val)
        val_loss_i = criterion(val_pred, labels_val)

        val_loss.append(train_loss_i)
        
        ave_val_loss = np.sum(val_loss) / len(val_loss)
        val_loss_list[idx] = ave_val_loss
        
    # validation for one epoch
    # val_pred = resNet(val_input.to(device))
    # val_loss_i = criterion(val_pred, val_label)
    # val_loss.append(val_loss_i)

    # train_accu_list[i] = train_accu
    # val_loss_list[i] = val_loss
    # val_accu_list[i] = val_accu
    save(idx, resNet, model_path)
    
    print('\n$Epoch: {}\t$training loss: {}\t$validation loss: {}'.format(idx + 1, ave_train_loss, ave_val_loss))
    print('Time: {}'.format(time.time() - start_time))
    print('-----------------------------------------------------------------------------------------')
    
    



# for i in range(n_epoch):
#     # first get a minibatch of data
#     train_loss = []
#     start_time = time.time()
#
#     for j in range(n_iteration):
#         #         print("\nbatch", j, ":")
#         #         start_time = time.time()
#
#         batch_start_index = j * batch_size
#         # get data batch from the normalized data
#         # X_batch = X_train_tensor[batch_start_index:batch_start_index + batch_size]
#         # get ground truth label y
#         # y_batch = y_train_tensor[batch_start_index:batch_start_index + batch_size]
#
#         #         print(X_batch.shape)
#         train_pred = resNet(X_batch)
#         #         print(train_pred.shape)
#         train_crt, train_accu = get_correct_and_accuracy(train_pred, y_batch)
#         train_loss_i = criterion(train_pred, y_batch)
#
#         #         train_loss.append(train_loss_i)
#         train_loss.append(train_loss_i.detach().numpy())
#
#         # Backpropagation
#         optimizer.zero_grad()
#         train_loss_i.backward()
#         optimizer.step()
#
#     #         print("batch", j, ":\t", time.time() - start_time)
#
#     #
#     val_pred = resNet(X_val_tensor)
#     val_crt, val_accu = get_correct_and_accuracy(val_pred, y_val_tensor)
#     val_loss = criterion(val_pred, y_val_tensor)
#
#     ave_train_loss = np.sum(train_loss) / len(train_loss)
#
#     print("Iter %d ,Train loss: %.3f, Train acc: %.3f, Val loss: %.3f, Val acc: %.3f"
#           % (i, ave_train_loss, train_accu, val_loss, val_accu))
#     ## add to the logs so that we can use them later for plotting
#     train_loss_list[i] = ave_train_loss
#     train_accu_list[i] = train_accu
#     val_loss_list[i] = val_loss
#     val_accu_list[i] = val_accu
#     print("iteration", i, ":\t", time.time() - start_time)