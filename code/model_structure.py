from model import ConvNet, wasteModel_CNN, ResNet, ResBlock
from config_training import config

image_size = config['image_size']

resNet = ResNet(ResBlock, img_size=image_size)
wasteCNN = wasteModel_CNN(image_size)
CNN = ConvNet(image_size)

net = resNet

print("model structure:", net)
