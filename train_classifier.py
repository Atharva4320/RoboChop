import numpy as np
import torch
import os, os.path
import math
from torchsummary import summary
from datetime import datetime
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torchivision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import matplotlib.image as mpimg

# train classifier on fruits360 dataset
# finetune classifier on our custom dataset

# ------ define hyperparameters --------
epochs = 100
batch_size = 512
img_height = 100
img_width = 100
n_channels = 3

# ------- import and process data -------
train_dir = ""
test_dir = ""

transform_data = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                     transforms.Resize(img_height, img_width)])

train_dataset = ImageFolder(train_dir, transform=transform_data)
test_dataset = ImageFolder(test_dir, trasnform=transform_data)
num_classes = len(train_dataset.classes)

dataloader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
dataloader_test = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

print('Train Dataset Size: ', len(train_dataset))
print('Test Dataset Size: ', len(test_dataset))

# ------- visualize some of the fruits -------
fig, ax = plt.subplots(1,4,figsize=(12, 9), dpi=120)
plt.setp(ax, xticks=[], yticks=[])
ax[0].imshow(mpimg.imread(train_dir+'/Apple Braeburn/0_100.jpg'))
ax[1].imshow(mpimg.imread(train_dir+'/Banana/0_100.jpg'))
ax[2].imshow(mpimg.imread(train_dir+'/Avocado ripe/0_100.jpg'))
ax[3].imshow(mpimg.imread(train_dir+'/Cherry 2/105_100.jpg'))
plt.show()

# ------- plot fruit/veg category counts -------
def plot_category_counts(path,xlabel,ylabel,title):
    categories = []
    counts = []
    for dir in os.listdir(path):
        categories.append(dir)
        counts.append(len(os.listdir(train_dir+"/"+ dir)))
    
    plt.rcParams["figure.figsize"] = (40,20)
    index = np.arange(len(categories))
    plt.bar(index, counts)
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.xticks(index, categories, fontsize=15, rotation=90)
    plt.title(title, fontsize=30)
    plt.show()

plot_category_counts(train_dir+"/",'Fruit Categories','Category Counts','Fruit Categories Training Distribution')
plot_category_counts(test_dir+"/",'Fruit Categories','Category Counts','Fruit Categories Testing Distribution')

# -------- define scheduler and early stopping -------
class LRScheduler():
    def __init__(self, optimizer, patience=5, min_lr=1e-7, factor=0.5):
        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,mode='min',patience=self.patience,factor=self.factor,min_lr=self.min_lr,verbose=True)
    def __call__(self, val_loss):
        self.lr_scheduler.step(val_loss)

class EarlyStopping():
    def __init__(self, patience=5, min_delta=0,save_best=False):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.save_best=save_best
    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
            if self.save_best:
                self.save_best_model()
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.save_best:
                self.save_best_model()
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
    def save_best_model(self, model):
        print(">>> Saving the current model with the best loss value...")
        print("-"*100)
        torch.save(model.state_dict(), 'best_loss_model.pth')

