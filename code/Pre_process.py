import torchvision
from torchvision import transforms
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import SubsetRandomSampler
import numpy as np
from config_training import config
import pandas as pd

'''
    Content of preprocessing
    1. convert to tensor
    2. Normalization
    3. Random transformation
    4. The resolution is changed to image_size * image_size

    all the other parameters can be changed in the global var (except the label)
'''

'''Define global variable'''
# dataset path
data_path = config['data_path']
# image size
image_size = 256        # 128

# label
dry_trash = 0
poison_trash = 1
recycle_trash = 2
wet_trash = 3

# Data split
test_percentage = 0.2
val_percentage = 0.25
batch_size = config['batch_size']

# Use CPU or GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
    transforms.RandomCrop([128, 128]),  # Crop randomly
    transforms.GaussianBlur(kernel_size=5, sigma=(10.0, 10.0)),  # Gaussian blur
    transforms.Grayscale(3)  # To gray
]
transform_random = transforms.RandomChoice(transform_random)
# Conventional processing and one of other processing
transform_normal = [
    transforms.Resize([128, 128]),  # Resetting image resolution
    transforms.ToTensor(),  # To [0, 1]
    transforms.Normalize([mean_r, mean_g, mean_b], [std_r, std_g, std_b]),
    transform_random
]
transform = transforms.Compose(transform_normal)

# load image data
data_all = torchvision.datasets.ImageFolder(root=data_path, transform=transform)



def dataset_sampler(dataset):
    """
    split dataset into train set, val set, test set
    :param dataset
    :return: split sampler
    """
    sample_num = len(dataset)
    file_idx = list(range(sample_num))
    train_idx, test_idx = train_test_split(file_idx, test_size=test_percentage, random_state=42)
    train_idx, val_idx = train_test_split(train_idx, test_size=val_percentage, random_state=42)
    # all_sampler = SubsetRandomSampler(file_idx)
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    return train_sampler, val_sampler, test_sampler


# get all the training, validation and test set here
train_sampler, val_sampler, test_sampler = dataset_sampler(data_all)
# loader = torch.utils.data.DataLoader(data_all, batch_size=batch_size, num_workers=0, sampler=all_sampler)
train_loader = torch.utils.data.DataLoader(data_all, batch_size=batch_size, num_workers=0, sampler=train_sampler)
val_loader = torch.utils.data.DataLoader(data_all, batch_size=batch_size, num_workers=0, sampler=val_sampler)
test_loader = torch.utils.data.DataLoader(data_all, sampler=test_sampler)


# how to use data loader
# for i, data in enumerate(train_loader):
#     inputs = data[0].to(device)
#     labels = data[1].to(device)
#     print(labels.shape)
#     for idx, batch in inputs:
#         dataframe = pd.DataFrame({'path': '', 'index': idx, 'label': labels[idx]})
#
#     dataframe.to_csv('train_data.csv', index=False, sep=',')
#
# for i, data in enumerate(test_loader):
#     inputs = data[0].to(device)
#     labels = data[1].to(device)
#     dataframe = pd.DataFrame({'index': inputs, 'label': labels})
#     dataframe.to_csv('test_data.csv', index=False, sep=',')
#
# for i, data in enumerate(val_loader):
#     inputs = data[0].to(device)
#     labels = data[1].to(device)
#     dataframe = pd.DataFrame({'index': inputs, 'label': labels})
#     dataframe.to_csv('val_data.csv', index=False, sep=',')

'''Calculate mean and std'''
'''
sum_r = 0
sum_g = 0
sum_b = 0
var_r = 0
var_g = 0
var_b = 0
for i in range(16016):
    r = inputs[i][0].numpy()
    g = inputs[i][1].numpy()
    b = inputs[i][2].numpy()
    sum_r += np.mean(r)
    sum_g += np.mean(g)
    sum_b += np.mean(b)
    var_r += np.var(r)
    var_g += np.var(g)
    var_b += np.var(b)


mean_r = sum_r / 16016
mean_g = sum_g / 16016
mean_b = sum_b / 16016
std_r = (var_r / 16016) ** 0.5
std_g = (var_g / 16016) ** 0.5
std_b = (var_b / 16016) ** 0.5

print(mean_r, mean_g, mean_b)
print(std_r, std_g, std_b)
'''
