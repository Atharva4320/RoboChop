import torch
import warnings
import sys
import torch
import torchvision
import cv2
import matplotlib.pyplot as plt
import numpy as np
import PIL
import time
from google.colab.patches import cv2_imshow
from tqdm import tqdm
import os
import warnings
import urllib.request
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import ultralytics
from ultralytics import YOLO


url = 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth'
filename = 'sam_vit_h_4b8939.pth'
urllib.request.urlretrieve(url, filename)

sam_checkpoint = "sam_vit_h_4b8939.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device.type == 'cuda':
  print('CUDA is found! Executing on %s.......'%torch.cuda.get_device_name(0))
else:
  warnings.warn('CUDA not found! Execution may be slow......')

model_type = "default"

sys.path.append("..")

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

mask_generator = SamAutomaticMaskGenerator(sam)

# Calculates the centroid of the object
def get_centroid(mask_segmentation):
  binary_mask = mask_segmentation.astype(np.uint8)

  y, x = np.indices(binary_mask.shape)
  centroid_x = int(np.sum(x * binary_mask) / np.sum(binary_mask))
  centroid_y = int(np.sum(y * binary_mask) / np.sum(binary_mask))

  return centroid_x, centroid_y
  
  
'''
TODO
'''
def disp_mask_and_centroid(image, anns, random_color=False, disp_centroid=True):
    if len(anns) == 0:
        return
  
     # Sort masks based on 'area' in descending order
    sorted_masks = sorted(anns, key=lambda x: x['area'], reverse=True)
    
    # Select point-of-interest mask
    poi_mask = sorted_masks[0]['segmentation']

    #Compute the centroid of the mask
    cent_x, cent_y = get_centroid(poi_mask)

    # Create an empty mask of zeros with the same shape as image
    mask_overlay = np.zeros(image.shape, dtype=np.uint8)
      
    # Convert the mask to uint8
    poi_mask = (poi_mask * 255).astype(np.uint8)

    # Find contours from the binary mask
    contours, _ = cv2.findContours(poi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Fill the poi_mask area with a specified color
    color = (255, 165, 0) if not random_color else tuple(np.random.randint(0, 255, 3).tolist())  # orange or random color
    cv2.drawContours(mask_overlay, contours, -1, color, thickness=cv2.FILLED)

    # Bitwise operation to put mask on top of image
    img = cv2.bitwise_and(image, mask_overlay)

    # If you want to show centroid, draw a small circle
    if disp_centroid:
        cv2.circle(img, (cent_x, cent_y), 5, (0, 255, 255), -1)  # Red color for centroid

        # Specify the font
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Specify the color
        color = (255, 0, 0)  # Red color

        # Specify the size
        size, thickness = 0.75, 2

        coordinates_text = f"({cent_x}, {cent_y})"
        cv2.putText(img, coordinates_text, (cent_x - 50, cent_y - 10), font, size, color, thickness)


    img_ = cv2.bitwise_not(img)

    # Create a mask where white (255) pixels in the overlay image are black (0) 
    mask = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY_INV)

    # Split the mask into 3 channels
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # Create a masked foreground and background
    foreground = cv2.bitwise_and(img_, mask)
    background = cv2.bitwise_and(image, cv2.bitwise_not(mask))

    # Combine the foreground and the background
    output_img = cv2.add(foreground, background)


    return cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB), cent_x, cent_y


ultralytics.checks()  # Check if YOLO is installed properly


model_path = '/content/drive/MyDrive/Fluid Segmentation Drive/yolov8n.pt'
YOLO = YOLO(model_path)  # load a pretrained YOLOv8n detection model


"""### Process the video:"""

'''
TODO
'''
def YOLO_SAM_Centroid(frame, model, poi='', yolo_centroid=True, sam_centroid=False, return_frame=True, 
                      display_mask=False, yolo_all=False):

    if yolo_all or poi=='': # Work here: Assume yolo_centroid +
          all_obj_img, x1, y1, x2, y2 = detect_objects(frame, model, detect_all=yolo_all)
          warnings.warn("Warning! No point-of-interest object mentioned in 'poi' field. Returning bounding boxes of all items.")
    
    og_img = frame.copy()
    start_time_yolo = time.time()
    # Perform object detection on the current frame
    cropped_img, x1, y1, x2, y2 = detect_objects(frame, model, poi)
    end_time_yolo = time.time()

    if x1 == 0:
      frame = cropped_img
      if return_frame:
          return frame, 0, 0
      else:
          return 0, 0

    else:
      if yolo_centroid: # Approximate and display the centroid using yolo bounding box
          # frame = cropped_img
          # Specify the color
          color = (0, 255, 0)  # Green color -> BGR

          # Specify the thickness
          thickness = 2

          # Specify the length of the lines of the cross
          line_length = 5  

          # Approximate the centroid using yolo bounding box
          yolo_centX = (x2 + x1)//2
          yolo_centY = (y2 + y1)//2

          # print("YOLO Approximation centroid: x={}, y={}\n".format(yolo_centX, yolo_centY))
          # Draw the first line of the cross (from top-left to bottom-right)
          cv2.line(frame, (yolo_centX - line_length, yolo_centY - line_length), 
                          (yolo_centX + line_length, yolo_centY + line_length), color, thickness)

          # Draw the second line of the cross (from top-right to bottom-left)
          cv2.line(frame, (yolo_centX + line_length, yolo_centY - line_length), 
                          (yolo_centX - line_length, yolo_centY + line_length), color, thickness)

          # Specify the font
          font = cv2.FONT_HERSHEY_SIMPLEX

          # Specify the size
          size, thickness = 0.75, 2

          coordinates_text = f"({yolo_centX}, {yolo_centY})"
          cv2.putText(frame, coordinates_text, (yolo_centX - 50, yolo_centY - 50), font, size, color, thickness)

          centroid_x, centroid_y = yolo_centX, yolo_centY

      elif sam_centroid:
          start_time_sam = time.time()
          cropped_mask = model.generate(cropped_img)
          end_time_sam = time.time()
          print("Time elapsed SAM: {}s".format(np.abs(end_time_sam - start_time_sam)))

          cropped_mask_img, centroid_x, centroid_y = disp_mask_and_centroid(cropped_img, cropped_mask, disp_centroid=cropped_centroid)

          if display_mask:
            frame[y1:y2, x1:x2] = cv2.cvtColor(cropped_mask_img, cv2.COLOR_RGB2BGR)

          ## Calculate actual centroids:
          # add centroid x to x_min (x1) of bounding box, add centroid y to the y_min (y1) of the bounding box
          sam_centX, sam_centY = centroid_x + x1, centroid_y + y1
          

          # Draw the actual centroids on the frame:
          color = (0,0,255)  # Red color -> BGR
          print("SAM Segmentation Centroid: x={}, y={}\n".format(sam_centX, sam_centY))

          # Display the actual centroid on the frame
          cv2.circle(frame, (sam_centX, sam_centY), radius=5, color=color, thickness=cv2.FILLED)

          # Specify the font
          font = cv2.FONT_HERSHEY_SIMPLEX

          # Specify the size
          size, thickness = 0.75, 2

          coordinates_text = f"({sam_centX}, {sam_centY})"
          cv2.putText(frame, coordinates_text, (sam_centX - 50, sam_centY - 10), font, size, color, thickness)

          centroid_x, centroid_y = sam_centX, sam_centY

      if return_frame:
          return frame, centroid_x, centroid_y
      else:
          return centroid_x, centroid_y

'''
TODO
'''
def detect_objects(image, model, target_class='',  detect_all=False, print_class_specifics=False):
  results = model.predict(image)

  if detect_all:
      img = results[0].plot()
      return img, 0, 0, 0, 0

  result = results[0]                                        
  boxes = result.boxes.cpu().numpy()  
  names = np.array([result.names[i] for i in boxes.cls])
  indices = np.where(names == target_class)  # Search for target object
  if len(indices[0]) != 0:  # Found a target object
      x1, y1, x2, y2 = boxes[indices[0][0]].xyxy[0].astype(int)  # Get the box coordinates of the target

      cropped_image = image[y1:y2, x1:x2]

      return image, x1, y1, x2, y2
      # return img, x1, y1, x2, y2

  else:
      return image, 0, 0, 0, 0
      

if __name__ == '__main__':
    # Load the video
    video_path = # Load the appropriate video path #'/content/drive/MyDrive/Fluid Semantic Drive/META [Segment Anything]/Videos/cam_1_video.mp4'
    video = cv2.VideoCapture(video_path)

    ## Flags
    yolo_all = False  #TODO

    output_path = # Load the appropriate video path #'/content/drive/MyDrive/Fluid Semantic Drive/META [Segment Anything]/YOLO_Detection/yolo_apple_centroid_cam_1_video.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter(output_path, fourcc, 30.0, (int(video.get(3)), int(video.get(4))))

    # Get total number of frames
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_counter = 0
    while video.isOpened():
        ret, frame = video.read()
        # Break the loop if the video ends
        if not ret:
            break
        result_frame, c_x, c_y = YOLO_SAM_Centroid(frame, YOLO, poi='apple', yolo_all=yolo_all)
        print(f"Object Centroid: ({c_x}, {c_y})") if c_x != 0 else None
        output_video.write(result_frame)
        frame_counter += 1

    print("Process Finished!!!")
    video.release()
    output_video.release()
    cv2.destroyAllWindows()


