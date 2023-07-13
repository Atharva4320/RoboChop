import os
import shutil

def extract_interested_items(base_directory, interested_class, max_count=0):
    # Define class names
    class_names = ['Apple', 'Cucumber', 'Lime', 'Orange', 'Pineapple']

    # Define directories and their corresponding ratios for the split
    sub_dirs = {"train": 0.7, "valid": 0.2, "test": 0.1}

    for sub_dir, ratio in sub_dirs.items():
        # Define the max count for this directory
        dir_max_count = int(max_count * ratio) if max_count != 0 else 0

        # Initialize counter for each directory
        count = 0

        # Define path for labels
        label_dir_path = os.path.join(base_directory, sub_dir, "labels")
        # List all files in the directory
        label_files = os.listdir(label_dir_path)

        # Loop over all files
        for file in label_files:
            # Stop if we already have max_count items
            if dir_max_count != 0 and count >= dir_max_count:
                break

            # Read the txt file
            with open(os.path.join(label_dir_path, file), 'r') as f:
                lines = f.readlines()
            
            # Extract class index and map it to class name
            for line in lines:
                class_index = int(line.split()[0])
                class_name = class_names[class_index]
                
                # Check if class name is equal to the interested class
                if class_name == interested_class:
                    # Define destination directories
                    dest_dir_label = os.path.join(base_directory, sub_dir, "interested_item", "labels")
                    dest_dir_image = os.path.join(base_directory, sub_dir, "interested_item", "images")

                    # Create destination directories if they don't exist
                    if not os.path.exists(dest_dir_label):
                        print(dest_dir_label, " folder does not exist! Creating...")
                        os.makedirs(dest_dir_label)
                    if not os.path.exists(dest_dir_image):
                        print(dest_dir_image, " folder does not exist! Creating...")
                        os.makedirs(dest_dir_image)

                    # Copy the label file
                    shutil.copy(os.path.join(label_dir_path, file), 
                                os.path.join(dest_dir_label, file))
                    
                    # Copy the corresponding image file
                    corresponding_image_file = file.replace('.txt', '.jpg')
                    shutil.copy(os.path.join(base_directory, sub_dir, "images", corresponding_image_file), 
                                os.path.join(dest_dir_image, corresponding_image_file))

                    # Increase counter
                    count += 1
                    # Break after copying the required number of images
                    if dir_max_count != 0 and count >= dir_max_count:
                        break

# Use the function
if __name__ == '__main__': 
    extract_interested_items("/Users/atharvasd/Downloads/Sliced Fruits and Vegetables-4", "Apple")
