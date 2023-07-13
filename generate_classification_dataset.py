import numpy as np
import os
import cv2
from tqdm import tqdm

# # ------- converts YOLOv8 dataset to classification dataset ---------
# label_dict = {0: 'Apple',
#           1: 'Carrot',
#           2: 'Cucumber',
#           3: 'Lime',
#           4: 'Orange',
#           5: 'Pineapple'}

# class_count = {0: 0,
#                 1: 0,
#                 2: 0,
#                 3: 0,
#                 4: 0,
#                 5: 0
# }

# data_path = '/home/alison/Desktop/YOLOCutFruit/test'
# save_path = '/home/alison/Desktop/ClassificationCutFruit/Test'
# img_path = data_path + '/images'

# for filename in tqdm(os.listdir(img_path)):
#     f = os.path.join(img_path, filename)
#     if os.path.isfile(f):
#         img = cv2.imread(f)
#         shape = img.shape
#         if shape[0] > 25 and shape[1] > 25:
#             str_len = len(filename)
#             file = data_path + '/labels/' + filename[0:(str_len-4)] + '.txt'
#             t = open(file, 'r')
#             text = t.read() # TODO: fix this!
#             label = int(text[0])
#             save_folder = save_path + '/' + label_dict[label] + '/'
#             name = str(class_count[label]) + '_100.jpg'
#             cv2.imwrite(save_folder + name, img)
#             class_count[label]+=1



# ---------- reformats Fruits360 dataset to more general classes ---------
label_dict = {
    'Apple Braeburn': 'Apple',
    'Cantaloupe 1': 'Cantaloupe',
    'Grape Blue': 'Grape',   
    'Pear Monster': 'Pear',   
    'Potato White': 'Potato',
    'Apple Crimson Snow': 'Apple',
    'Cantaloupe 2': 'Cantaloupe',
    'Pear Red': 'Pear',            
    'Apple Golden 1': 'Apple',        
    'Apple Golden 2': 'Apple',
    'Grape Pink': 'Grape',         
    'Pear Williams': 'Pear',
    'Apple Golden 3': 'Apple',
    'Cherry 1': 'Cherry',
    'Grape White': 'Grape',
    'Apple Granny Smith': 'Apple', 
    'Cherry 2': 'Cherry',
    'Grape White 2': 'Grape',
    'Apple Pink Lady': 'Apple',
    'Grape White 3': 'Grape',
    'Apple Red 1': 'Apple',
    'Cherry Wax Black': 'Cherry',
    'Grape White 4': 'Grape',
    'Strawberry Wedge': 'Strawberry',
    'Strawberry': 'Strawberry',
    'Apple Red 2': 'Apple',
    'Cherry Wax Red': 'Cherry',         
    'Onion Red': 'Onion',
    'Apple Red 3': 'Apple',  
    'Cherry Wax Yellow': 'Cherry',      
    'Onion Red Peeled': 'Onion',
    'Apple Red Delicious': 'Apple',
    'Onion White': 'Onion',
    'Tomato 1': 'Tomato',
    'Apple Red Yellow 1': 'Apple',             
    'Pineapple': 'Pineapple',
    'Tomato 2': 'Tomato',
    'Apple Red Yellow 2': 'Apple',                
    'Pineapple Mini': 'Pineapple',
    'Tomato 3': 'Tomato',
    'Corn': 'Corn',           
    'Tomato 4': 'Tomato',
    'Avocado': 'Avocado',  
    'Corn Husk': 'Corn',   
    'Peach': 'Peach',                 
    'Plum': 'Plum',      
    'Tomato Cherry Red': 'Cherry',
    'Avocado ripe': 'Avocado',
    'Cucumber Ripe': 'Cucumber',
    'Lemon': 'Lemon',
    'Peach 2': 'Peach',     
    'Plum 2': 'Plum',   
    'Tomato Heart': 'Tomato',
    'Banana': 'Banana',
    'Cucumber Ripe 2': 'Cucumber',  
    'Lemon Meyer': 'Lemon',
    'Peach Flat': 'Peach',
    'Plum 3': 'Plum', 
    'Tomato Maroon': 'Tomato',
    'Banana Lady Finger': 'Banana',         
    'Pear': 'Pear',        
    'Tomato not Ripened': 'Tomato',
    'Banana Red': 'Banana',                
    'Pear 2': 'Pear',
    'Tomato Yellow': 'Tomato',
    'Pear Abate': 'Pear',
    'Potato Red': 'Potato',
    'Mango': 'Mango',         
    'Pear Forelle': 'Pear',      
    'Potato Red Washed': 'Potato',
    'Mango Red': 'Mango',
    'Pear Kaiser': 'Pear'
}

file_counts = {
    'Apple': 0,
    'Cantaloupe': 0,
    'Grape': 0,
    'Pear': 0,
    'Potato': 0,
    'Strawberry': 0,
    'Cherry': 0,
    'Onion': 0,
    'Tomato': 0,
    'Pineapple': 0,
    'Corn': 0,
    'Peach': 0,
    'Lemon': 0,
    'Avocado': 0,
    'Cucumber': 0,
    'Banana': 0,
    'Mango': 0,
}

dataset_path = '/home/alison/Desktop/Fruit360/fruits-360_dataset/fruits-360/Training'
new_path = '/home/alison/Desktop/Fruit360/fruits-360_dataset/fruits-360/General_Classes/Training'

for folder in tqdm(os.listdir(dataset_path)):
    if folder in label_dict:
        new_label = label_dict[folder]
        # if new_label folder doesn't exist at new_path, make the folder
        # iterate through folder
            # import image
            # rename based on the count for new_label
            # save to new folder
            # file_counts[new_label]+=1

# iterate through all of the folders of fruits360
    # if the folder is within a specific dictionary, add it to the dictionary value folder instead
    # else copy the whole folder over as is

