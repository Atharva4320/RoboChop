import numpy as np

# given the yolov8 custom dataset iterate through the images
    # iterate through the bounding boxes for each image
        # crop the image with the bounding box and save --> cropped_image = frame[y1:y2, x1:x2]
        # save the label associated with that bounding box
