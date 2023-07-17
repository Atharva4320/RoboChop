import numpy as np
import torch
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

# train classifier on fruits360 dataset
# finetune classifier on our custom dataset

# ------ define hyperparameters --------
epochs = 100
batch_size = 512
img_height = 100
img_width = 100
n_channels = 3
pre_train = False
pre_train_path = 'best_loss_model.pth'
save_path = 'custom_nopretrain.pth'

# ------- import and process data -------
if pre_train:
    train_dir = "/home/alison/Desktop/ClassificationCutFruit/Training"
    test_dir = "/home/alison/Desktop/ClassificationCutFruit/Test"
else:
    train_dir = "/home/alison/Desktop/ClassificationCutFruit/Training"
    test_dir = "/home/alison/Desktop/ClassificationCutFruit/Test"
    # train_dir = "/home/alison/Desktop/Fruit360/fruits-360_dataset/fruits-360/General_Classes/Training"
    # test_dir = "/home/alison/Desktop/Fruit360/fruits-360_dataset/fruits-360/General_Classes/Test"
    

transform_data = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                     transforms.Resize((img_height, img_width), antialias=True)])

train_dataset = ImageFolder(train_dir, transform=transform_data)
test_dataset = ImageFolder(test_dir, transform=transform_data)
num_classes = len(train_dataset.classes)
print("\nNum Classes: ", num_classes)

dataloader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
dataloader_test = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

print('Train Dataset Size: ', len(train_dataset))
print('Test Dataset Size: ', len(test_dataset))

# # ------- visualize some of the fruits -------
# fig, ax = plt.subplots(1,4,figsize=(12, 9), dpi=120)
# plt.setp(ax, xticks=[], yticks=[])
# ax[0].imshow(mpimg.imread(train_dir+'/Apple Braeburn/0_100.jpg'))
# ax[1].imshow(mpimg.imread(train_dir+'/Banana/0_100.jpg'))
# ax[2].imshow(mpimg.imread(train_dir+'/Avocado ripe/0_100.jpg'))
# ax[3].imshow(mpimg.imread(train_dir+'/Cherry 2/105_100.jpg'))
# plt.show()

# # ------- plot fruit/veg category counts -------
# def plot_category_counts(path,xlabel,ylabel,title):
#     categories = []
#     counts = []
#     for dir in os.listdir(path):
#         categories.append(dir)
#         counts.append(len(os.listdir(train_dir+"/"+ dir)))
    
#     plt.rcParams["figure.figsize"] = (40,20)
#     index = np.arange(len(categories))
#     plt.bar(index, counts)
#     plt.xlabel(xlabel, fontsize=20)
#     plt.ylabel(ylabel, fontsize=20)
#     plt.xticks(index, categories, fontsize=15, rotation=90)
#     plt.title(title, fontsize=30)
#     plt.show()

# plot_category_counts(train_dir+"/",'Fruit Categories','Category Counts','Fruit Categories Training Distribution')
# plot_category_counts(test_dir+"/",'Fruit Categories','Category Counts','Fruit Categories Testing Distribution')

model = Fruits_CNN(num_channels=n_channels, num_classes=num_classes) #62)
if pre_train:
    model.load_state_dict(torch.load(pre_train_path))

model = model.cuda()

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
lr_scheduler = LRScheduler(optimizer= optimizer,patience=5,min_lr=1e-7, factor=0.5)
early_stopping = EarlyStopping(patience=15, min_delta=0, save_best=True)

# print model summary
print(summary(model, (n_channels, img_height,img_width),batch_size))
    
train_loss = []
test_loss = []
train_accuracy = []
test_accuracy = []

print('Start Training')
print('*'*100)

for epoch in range(epochs):
    start_time = datetime.now()

    # TRAINING
    correct = 0
    iterations = 0
    iter_loss = 0.0

    model.train()
    for i, (inputs, labels) in enumerate(dataloader_train):
        inputs = inputs.cuda()
        labels = labels.cuda()

        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        iter_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum()
        iterations += 1

    train_loss.append(iter_loss / iterations)
    train_accuracy.append(100 * correct / len(train_dataset))

    # TESTING
    loss_testing = 0.0
    correct = 0
    iterations = 0

    model.eval()

    for i, (inputs, labels) in enumerate(dataloader_test):
        inputs = inputs.cuda()
        labels = labels.cuda()

        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss_testing += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum()

        iterations += 1

    test_loss.append(loss_testing / iterations)
    test_accuracy.append(100 * correct / len(test_dataset))

    print('Epoch {}/{}, Training Loss: {:.3f}, Training Accuracy: {:.3f}, Testing Loss: {:.3f}, Testing Acc: {:.3f}'
          .format(epoch + 1, epochs, train_loss[-1], train_accuracy[-1], test_loss[-1], test_accuracy[-1]))

    end_time = datetime.now()
    epoch_time = (end_time - start_time).total_seconds()
    print("-"*100)
    print('Epoch Time : ', math.floor(epoch_time // 60), ':', math.floor(epoch_time % 60))
    print("-"*100)

    lr_scheduler(test_loss[-1])
    early_stopping(test_loss[-1], model, save_path)
    if early_stopping.early_stop:
        print('*** Early stopping ***')
        break

print("*** Training Completed ***")
