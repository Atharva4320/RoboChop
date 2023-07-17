import numpy as np
import torch
import cv2
import os, os.path
import math
from torchsummary import summary
from datetime import datetime
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import matplotlib.image as mpimg
from classifier_model  import *

pre_train_path = 'custom_nopretrain.pth'

img_height = 100
img_width = 100
n_channels = 3
transform_data = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                     transforms.Resize((img_height, img_width), antialias=True)])

model = Fruits_CNN(num_channels=n_channels, num_classes=6) #74)
model.load_state_dict(torch.load(pre_train_path))
model = model.cuda()
model.eval()


# -------- testing on real-world data ----------
label_dict = {0: 'Apple',
              1: 'Carrot',
              2: 'Cucumber',
              3: 'Lime',
              4: 'Orange',
              5: 'Pineapple'}

images = cv2.imread("/home/alison/Desktop/Atharva/merged_image_7.jpg")# import image to process
plt.imshow(images)
plt.show()

images = transform_data(images)
images = torch.unsqueeze(images, dim=0)
print(images.size())
images = images.cuda()
images = images.float()

outputs = model(images)
_, predicted = torch.max(outputs, 1)
predicted = predicted.cpu().detach().numpy()[0]

print("Predicted Number: ", predicted)
print("Predicted Class: ", label_dict[predicted])