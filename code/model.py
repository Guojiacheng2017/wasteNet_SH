# created by Jiacheng Guo at Dec 4 15:22:58 CST 2021
# ResNet --jupyter version
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from config_training import config
batch_size = config['batch_size']


#--------------------------------------------------------------------------------------------------#

class ConvNet(nn.Module):
    def __init__(self, img_size):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3) # nn.Sequential()
        self.relu = nn.ReLU()
        self.padding = nn.ZeroPad2d(1)
        self.fc1 = nn.Linear(4 * img_size * img_size, 128)
        self.fc2 = nn.Linear(128, 84)
        self.fc3 = nn.Linear(84, 10)
        self.flat = nn.Flatten(1)
        self.max_pool = nn.MaxPool2d(2)

    def forward(self, x):
        covlayer1 = self.max_pool(self.relu(self.conv1(self.padding(x)))) # size * size -> size/2 * size/2
        covlayer2 = self.max_pool(self.relu(self.conv2(self.padding(covlayer1)))) # size/2 * size/2 -> size/4 * size/4
#         x = torch.flatten(covlayer2, 1)
#         covlayer2.reshape(covlayer2.shape[0], -1)
        covlayer2 = self.flat(covlayer2)
#         print(covlayer2.shape)
        x = self.relu(self.fc1(covlayer2))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


    
# CNN
#--------------------------------------------------------------------------------------------------#
class wasteModel_CNN(nn.Module):
    def __init__(self, img_size):
        super(wasteModel_CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)  # nn.Sequential()
        self.convC = nn.Conv2d(64, 128, 3)
        self.relu = nn.ReLU()
        self.zero_padding = nn.ZeroPad2d(1)
        self.miro_padding = nn.ReflectionPad2d(1)
        self.fc1 = nn.Linear(img_size*img_size*2, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 4)
        self.flat = nn.Flatten(1)
        self.max_pool = nn.MaxPool2d(2)

    def forward(self, x):
        conv_layer1 = self.max_pool(self.relu(self.conv1(self.zero_padding(x))))
        conv_layer2 = self.max_pool(self.relu(self.conv2(self.zero_padding(conv_layer1))))
        conv_layer3 = self.max_pool(self.relu(self.convC(self.zero_padding(conv_layer2))))
        conv_flat = self.flat(conv_layer3)
#         print(conv_flat.shape)
        fcl_rst = self.fc3(self.relu(self.fc2(self.relu(self.fc1(conv_flat)))))
        return fcl_rst


    
# Resnet
#--------------------------------------------------------------------------------------------------#

class ResBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            # shortcut, make sure the input and output size is matched
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, X):
        body = self.block(X)
        # print(body.shape, self.shortcut(X).shape)
        body = body + self.shortcut(X)
        body = F.relu(body)
        return body

    

class ResNet(nn.Module):
    def __init__(self, ResBlock, img_size=128, num_classes=4):
        super(ResNet, self).__init__()
        # img.shape = 1 here
        self.inchannel = 64

        self.conv = nn.Sequential(
            nn.Conv2d(3, self.inchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.inchannel),
            nn.ReLU()
        ) # keep the original size
        self.layer1 = self.make_layer(ResBlock, 64, 2, stride=1) # keep size, input c = 64, out c = 128
        self.layer2 = self.make_layer(ResBlock, 128, 2, stride=2)# be quarter size / 4, input c = 128, out c = 256
        self.layer3 = self.make_layer(ResBlock, 256, 2, stride=2)# be quarter size / 4, input c = 256, out c = 512
        self.layer4 = self.make_layer(ResBlock, 512, 2, stride=2)# be quarter size / 4, input c = 512, out c = 1024
#         print(self.layer4)
        self.avgPool = nn.AvgPool2d(4) # / 16
        # self.fc_pre = nn.Linear((img_size/8 * img_size/ 8)/16 * 1024, ___)
#         print(int(img_size*img_size/1024))
#         self.fc = nn.Linear(256, num_classes)   # img_size = 32
        self.fc = nn.Linear(int(img_size*img_size*512/1024), num_classes) # 512 channels / (2^5 * 2^5) for each dimension

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, X):
        out = self.conv(X)
        # print("conv:", out.shape)
        out = self.layer1(out)
        # print("layer1:", out.shape)
        out = self.layer2(out)
        # print("layer2:", out.shape)
        out = self.layer3(out)
        # print("layer3:", out.shape)
        out = self.layer4(out)
        # print("layer4:", out.shape)
        out = self.avgPool(out)
        # print("avgPool:", out.shape)
        out = out.view(out.size(0), -1)
#         out = out.reshape(batch_size, -1)
        out = self.fc(out)
        return out

# def Res18():
#     return ResNet(ResBlock, num_classes=10)

#--------------------------------------------------------------------------------------------------#
