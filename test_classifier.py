import numpy as np
import torch
import cv2
import os, os.path
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from classifier_model import Fruits_CNN

pre_train_path = 'customdata_pretrain.pth' # 'general_fruits360.pth' # 'customdata_pretrain.pth' # 'custom_nopretrain.pth'

img_height = 100
img_width = 100
n_channels = 3
device = torch.device('cuda')
transform_data = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                     transforms.Resize((img_height, img_width), antialias=True)])

model = Fruits_CNN(num_channels=n_channels, num_classes=74) # 6) #74)
model.load_state_dict(torch.load(pre_train_path))
model = model.to(device)
model.eval()


# -------- testing on real-world data ----------
# label_dict = {0: 'Apple',
#               1: 'Carrot',
#               2: 'Cucumber',
#               3: 'Lime',
#               4: 'Orange',
#               5: 'Pineapple'}

label_dict = {0: 'Apple',
              1: 'Apricot',
              2: 'Avocado',
              3: 'Banana',
              4: 'Beetroot',
              5: 'Blueberry',
              6: 'Cactus fruit',
              7: 'Cantaloupe',
              8: 'Carambula',
              9: 'Cauliflower',
              10: 'Cherry',
              11: 'Chstnut',
              12: 'Clementine',
              13: 'Cocos',
              14: 'Corn',
              15: 'Cucumber',
              16: 'Dates',
              17: 'Eggplant',
              18: 'Fig',
              19: 'Ginger Root',
              20: 'Granadilla',
              21: 'Grape',
              22: 'Grapefruit',
              23: 'Guava',
              24: 'Hazelnut',
              25: 'Huckleberry',
              26: 'Kaki',
              27: 'Kiwi',
              28: 'Kohlrabi',
              29: 'Kumquats',
              30: 'Lemon',
              31: 'Lime',
              32: 'Lychee',
              33: 'Mandarine',
              34: 'Mango',
              35: 'Mangostan',
              36: 'Maracuja',
              37: 'Melon Piel de Sapo',
              38: 'Mulberry',
              39: 'Nectarine',
              40: 'Nut Forest',
              41: 'Nut Pecan',
              42: 'Onion',
              43: 'Orange',
              44: 'Papaya',
              45: 'Passion Fruit',
              46: 'Peach',
              47: 'Pear',
              48: 'Pear Stone',
              49: 'Pepino',
              50: 'Pepper Green',
              51: 'Pepper Orange',
              52: 'Pepper Red',
              53: 'Pepper Yellow',
              54: 'Physalis',
              55: 'Physalis with Husk',
              56: 'Pineapple',
              57: 'Pitahaya Red',
              58: 'Plum',
              59: 'Pomegranate',
              60: 'Pomelo Sweetie',
              61: 'Potato',
              62: 'Potato Sweet',
              63: 'Quince',
              64: 'Rhambutan',
              65: 'Rasperry',
              66: 'Redcurrant',
              67: 'Salak',
              68: 'Strawberry',
              69: 'Tamarillo',
              70: 'Tangelo',
              71: 'Tomato',
              72: 'Walnut',
              73: 'Watermelon'}

images = cv2.imread("/home/alison/Desktop/Atharva/SAM-ChatGPT_for_Kitchen_Tasks/merged_image_72.jpg")# import image to process
plt.imshow(images)
plt.show()

images = transform_data(images)
images = torch.unsqueeze(images, dim=0)
images = images.to(device)
images = images.float()
print("image shape: ", images.size())

outputs = model(images)
_, predicted = torch.max(outputs, 1)
predicted = predicted.cpu().detach().numpy()[0]
string_pred = label_dict[predicted]

print("Predicted Number: ", predicted)
print("Predicted Class: ", label_dict[predicted])