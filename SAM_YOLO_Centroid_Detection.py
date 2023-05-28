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
import os
import warnings
import urllib.request
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import ultralytics
from ultralytics import YOLO


def show_progress(count, block_size, total_size):
    """
    This function displays the progress of a download operation.

    Parameters:
    ----------
    count : int
        The current number of blocks transferred.
        
    block_size : int
        The size of a block. In most cases, this would be in bytes.
        
    total_size : int
        The total size of the file being downloaded.

    Returns:
    -------
    None
        This function doesn't return anything. It just prints the current progress of the download operation.
    """
   completed = count * block_size
   progress = completed / total_size * 100
   print("\rDownload progress: %d%%" % (progress), end=" ")
   


# Calculates the centroid of the object
def get_centroid(mask_segmentation):
    """
    This function calculates the centroid of the object based on the provided binary mask segmentation.

    Parameters:
    ----------
    mask_segmentation : ndarray
        The binary mask segmentation of the object. The object's area should be marked with 1s and the rest with 0s.
        It should be a 2-dimensional numpy array.

    Returns:
    -------
    tuple
        The x and y coordinates of the object's centroid.
    """
  binary_mask = mask_segmentation.astype(np.uint8)

  y, x = np.indices(binary_mask.shape)
  centroid_x = int(np.sum(x * binary_mask) / np.sum(binary_mask))
  centroid_y = int(np.sum(y * binary_mask) / np.sum(binary_mask))

  return centroid_x, centroid_y
  

def generate_SAM_centroid(image, anns, random_color=False, disp_centroid=False):
    """
    Generate the centroid of the target object in the image based on SAM (Segment Anything Model) segmentation.

    This function finds the segmentation mask with the largest area, computes its centroid, fills the mask
    with a color, and overlays it onto the original image. If desired, it will display the
    centroid on the image. Finally, it inverts the image, creates a mask from the white pixels,
    and combines the masked foreground and the background.

    Parameters:
    image (numpy.ndarray): The original image.
    anns (list): List of annotations, where each annotation is a dictionary containing 'area' and 'segmentation' as keys.
    random_color (bool, optional): If True, the color of the mask will be random. If False, the color will be orange. Default is False.
    disp_centroid (bool, optional): If True, the centroid of the mask will be displayed on the image. Default is False.

    Returns:
    tuple: A 3-tuple containing:
        - output_img (numpy.ndarray): The image with the mask overlaid.
        - cent_x (int): The x-coordinate of the centroid.
        - cent_y (int): The y-coordinate of the centroid.

    """
    if len(anns) == 0: return
    
    # Compute the centroid of the largest mask
    cent_x, cent_y = get_centroid(sorted(anns, key=lambda x: x['area'], reverse=True)[0]['segmentation'])
    
    # Convert the mask to uint8 and find contours
    contours, _ = cv2.findContours((sorted_masks[0]['segmentation'] * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a filled colored mask and apply it to the image
    color = (255, 165, 0) if not random_color else tuple(np.random.randint(0, 255, 3).tolist())
    mask_overlay = cv2.drawContours(np.zeros(image.shape, dtype=np.uint8), contours, -1, color, thickness=cv2.FILLED)
    img = cv2.bitwise_and(image, mask_overlay)
    
    if disp_centroid:
        # Display centroid position on image
        cv2.circle(img, (cent_x, cent_y), 5, (0, 255, 255), -1)  # Red color for centroid
        cv2.putText(img, f"({cent_x}, {cent_y})", (cent_x - 50, cent_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
    
    # Invert image and create mask from white pixels
    img_ = cv2.bitwise_not(img)
    mask = cv2.threshold(cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY), 254, 255, cv2.THRESH_BINARY_INV)[1]
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    # Combine the masked foreground and the background
    output_img = cv2.add(cv2.bitwise_and(img_, mask), cv2.bitwise_and(image, cv2.bitwise_not(mask)))
    
    return cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB), cent_x, cent_y



def detect_objects(image, model, target_class='', detect_all=False, print_class_specifics=False):
    """
    This function detects objects in a given image using a provided model. It allows for
    specific class detection or can detect all classes. It also provides an option to print
    class specifics if required.

    Parameters:
    ----------
    image : ndarray
        The image in which objects are to be detected. It should be a numpy array.
        
    model : Object
        The model to be used for prediction. It should have a predict() method which returns a
        Result object with attributes 'boxes' (coordinates of bounding boxes), 'names' (class
        names of detected objects), and 'plot()' method (to plot detected objects on the image).
        
    target_class : str, optional
        The class of the object to be detected. If not provided or empty, function will detect
        all objects in the image (default is '').
        
    detect_all : bool, optional
        A flag to indicate if all objects in the image are to be detected. If True, function will
        return the image with all objects detected and plotted (default is False).
        
    print_class_specifics : bool, optional
        A flag to indicate if class-specific details should be printed. This is not currently
        implemented in the function and won't affect function behavior (default is False).

    Returns:
    -------
    tuple
        If detect_all is True:
            Returns the image with all objects detected and plotted, and four zeros.
        Else:
            If target object is found, returns the original image and the bounding box coordinates.
            If target object is not found, returns the original image and four zeros.
    """

    # Perform object detection on the image using the model
    results = model.predict(image)

    # If detect_all flag is not set, proceed with finding the target_class
    result = results[0]
    
    # If detect_all flag is set to True, plot all detected objects on the image and return
    if detect_all:
        img = result.plot()
        return img, 0, 0, 0, 0

    boxes = result.boxes.cpu().numpy()  # Extract bounding box coordinates and convert them to numpy array
    names = np.array([result.names[i] for i in boxes.cls])  # Extract class names for detected objects

    # Find indices of the detected objects that match the target_class
    indices = np.where(names == target_class)  # Search for target object

    # If target_class is found in the detected objects, extract its bounding box coordinates
    if len(indices[0]) != 0:  # Found a target object
        x1, y1, x2, y2 = boxes[indices[0][0]].xyxy[0].astype(int)  # Get the box coordinates of the target
        return image, x1, y1, x2, y2
    else:  # If target_class is not found, return the original image and four zeros
        return image, 0, 0, 0, 0

   

def calculate_centroid(frame, model, poi='', yolo_centroid=False, sam_centroid=False, return_frame=True, display_mask=False, yolo_all=False):
    """
    This function calculates the centroid of detected objects in the given frame.

    Parameters:
    ----------
    frame : ndarray
        The input image frame. It should be a numpy array.
        
    model : Object
        The YOLO model to use for detection.
        
    poi : str, optional
        Point of Interest. Default is empty string.
        
    yolo_centroid : bool, optional
        If True, use YOLO for centroid calculation. Default is True.
        
    sam_centroid : bool, optional
        If True, use SAM for centroid calculation. Default is False.
        
    return_frame : bool, optional
        If True, returns the frame along with centroids. Default is True.
        
    display_mask : bool, optional
        If True, displays the mask. Default is False.
        
    yolo_all : bool, optional
        If True, YOLO is used for all objects. Default is False.

    Returns:
    -------
    tuple
        If return_frame is True, returns the frame with centroids and the centroids' coordinates.
        Otherwise, returns only the centroids' coordinates.
    """
    # Detect objects in frame
    if yolo_all or poi == '':
        frame, x1, y1, x2, y2 = handle_yolo_all(frame, model, yolo_all, poi, return_frame)
    else:
        frame, x1, y1, x2, y2 = detect_objects(frame, model, target_class=poi)

    # Handle zero coordinates
    if x1 == 0:
        return handle_zero_coordinates(frame, return_frame)

    # Calculate centroid based on the method selected
    if yolo_centroid:
        frame, centroid_x, centroid_y = calculate_yolo_centroid(frame, x1, y1, x2, y2)
    elif sam_centroid:
        try:
            frame, centroid_x, centroid_y = calculate_sam_centroid(frame, SAM, x1, y1, x2, y2, display_mask)
        except Exception:
            print("SAM model is not loaded properly. Please make sure that SAM is properly loaded.")

    return frame, centroid_x, centroid_y if return_frame else centroid_x, centroid_y



def handle_yolo_all(frame, model, yolo_all, poi, return_frame):
    """
    This function handles the 'yolo_all' condition during object detection.

    Parameters:
    ----------
    frame : ndarray
        The input image frame. It should be a numpy array.
        
    model : Object
        The YOLO model to use for detection.
        
    yolo_all : bool
        If True, YOLO is used for all objects.
        
    poi : str
        Point of Interest.
        
    return_frame : bool
        If True, returns the frame along with coordinates.

    Returns:
    -------
    tuple
        The frame and coordinates of the detected object.
    """
    # This function handles yolo_all condition
    frame, x1, y1, x2, y2 = detect_objects(frame, model, detect_all=yolo_all)
    if poi == '':
        warnings.warn("Warning! No point-of-interest object mentioned in 'poi' field. Returning bounding boxes of all items.")
    return frame, x1, y1, x2, y2



def handle_zero_coordinates(frame, return_frame):
    """
    This function handles the case where the detected object's x-coordinate is zero.

    Parameters:
    ----------
    frame : ndarray
        The input image frame. It should be a numpy array.
        
    return_frame : bool
        If True, returns the frame along with coordinates.

    Returns:
    -------
    tuple
        If return_frame is True, returns the frame and zero coordinates.
        Otherwise, returns only zero coordinates.
    """
    # This function handles the condition where x1 is zero
    return (frame, 0, 0) if return_frame else (0, 0)



def calculate_yolo_centroid(frame, x1, y1, x2, y2):
    """
    This function calculates the centroid using YOLO bounding box and draws it on the given frame.

    Parameters:
    ----------
    frame : ndarray
        The input image frame. It should be a numpy array.
        
    x1 : int
        The x-coordinate of the top-left corner of the bounding box.
        
    y1 : int
        The y-coordinate of the top-left corner of the bounding box.
        
    x2 : int
        The x-coordinate of the bottom-right corner of the bounding box.
        
    y2 : int
        The y-coordinate of the bottom-right corner of the bounding box.

    Returns:
    -------
    tuple
        The centroid's x and y coordinates.
    """
    # This function calculates the centroid using yolo bounding box
    yolo_centX, yolo_centY = (x2 + x1) // 2, (y2 + y1) // 2
    frame = draw_cross_centroid(frame, yolo_centX, yolo_centY, (0, 255, 0))
    return frame, yolo_centX, yolo_centY



def calculate_sam_centroid(frame, mask_generator, x1, y1, x2, y2, display_mask):
    """
    This function calculates the centroid using SAM (Sharpness-Aware Minimization for Efficiently Improving Generalization)
    and draws it on the given frame. It also has an option to display the generated mask.

    Parameters:
    ----------
    frame : ndarray
        The input image frame. It should be a numpy array.
        
    mask_generator : Object
        The SAM model to be used for centroid calculation.
        
    x1 : int
        The x-coordinate of the top-left corner of the bounding box.
        
    y1 : int
        The y-coordinate of the top-left corner of the bounding box.
        
    x2 : int
        The x-coordinate of the bottom-right corner of the bounding box.
        
    y2 : int
        The y-coordinate of the bottom-right corner of the bounding box.
        
    display_mask : bool
        If True, displays the mask.

    Returns:
    -------
    tuple
        The calculated centroid's x and y coordinates.
    """
    # This function calculates the centroid using sam method

    cropped_img = frame[y1:y2, x1:x2]
    cropped_mask = mask_generator.generate(cropped_img)
    cropped_mask_img, centroid_x, centroid_y = generate_SAM_centroid(cropped_img, cropped_mask)
    if display_mask:
        frame[y1:y2, x1:x2] = cv2.cvtColor(cropped_mask_img, cv2.COLOR_RGB2BGR)
    sam_centX, sam_centY = centroid_x + x1, centroid_y + y1
    frame = draw_circle_centroid(frame, sam_centX, sam_centY, (0, 0, 255))
    return frame, sam_centX, sam_centY



def draw_cross_centroid(frame, centX, centY, color):
    """
    This function draws cross centroid on the given frame.

    Parameters:
    ----------
    frame : ndarray
        The input image frame. It should be a numpy array.
        
    centX : int
        The x-coordinate of the centroid.
        
    centY : int
        The y-coordinate of the centroid.
        
    color : tuple
        A tuple representing BGR color values.

    Returns:
    -------
    ndarray
        The frame with the drawn centroid.
    """
    # This function draws cross centroid on the frame
    line_length, thickness = 5, 2
    cv2.line(frame, (centX - line_length, centY - line_length), (centX + line_length, centY + line_length), color, thickness)
    cv2.line(frame, (centX + line_length, centY - line_length), (centX - line_length, centY + line_length), color, thickness)
    font, size = cv2.FONT_HERSHEY_SIMPLEX, 0.75
    coordinates_text = f"({centX}, {centY})"
    cv2.putText(frame, coordinates_text, (centX - 50, centY - 50), font, size, color, thickness)
    return frame



def draw_circle_centroid(frame, centX, centY, color):
    """
    This function draws circle centroid on the given frame.

    Parameters:
    ----------
    frame : ndarray
        The input image frame. It should be a numpy array.
        
    centX : int
        The x-coordinate of the centroid.
        
    centY : int
        The y-coordinate of the centroid.
        
    color : tuple
        A tuple representing BGR color values.

    Returns:
    -------
    ndarray
        The frame with the drawn centroid.
    """
    # This function draws circle centroid on the frame
    cv2.circle(frame, (centX, centY), radius=5, color=color, thickness=cv2.FILLED)
    font, size = cv2.FONT_HERSHEY_SIMPLEX, 0.75
    coordinates_text = f"({centX}, {centY})"
    cv2.putText(frame, coordinates_text, (centX - 50, centY - 10), font, size, color, thickness)
    return frame



if __name__ == '__main__':

    #============= Loading the SAM Model =======================
    url = 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth'
    filename = 'sam_vit_h_4b8939.pth'

    model_path_SAM = os.path.join('Models', 'sam_vit_l_0b3195.pth')
    print(model_path_SAM)
    
    if os.path.isfile(model_path_SAM):
        print("SAM model file exists!")
        model_type = "default"

        sam = sam_model_registry[model_type](checkpoint=model_path_SAM)
        sam.to(device=device)

        SAM = SamAutomaticMaskGenerator(sam)
    else:
        print("The file does not exits.")
        
#    if os.path.isfile("sam_vit_h_4b8939.pth"):
#        print("File already exists!")
#
#    else:
#        urllib.request.urlretrieve(url, filename, reporthook=show_progress)
#        print("\nDownload complete!")
        
#    sam_checkpoint = "sam_vit_h_4b8939.pth"
    
    #============= Loading the YOLO Model =======================
    print("Check if YOLO properly installed:")
    ultralytics.checks()  # Check if YOLO is installed properly

    directory = '/home/master_students/Atharva/SAM-ChatGPT_for_Kitchen_Tasks/Models'
    model_path_YOLO = os.path.join('Models', 'yolov8n.pt')
    print(model_path_YOLO)

    if os.path.isfile(model_path_YOLO):
        print("YOLO model file exists!")
        YOLO = YOLO(model_path_YOLO)  # load a pretrained YOLOv8n detection model
    else:
        print("The file does not exits.")
    
    #============= Checking for cuda =======================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == 'cuda':
      print('CUDA is found! Executing on %s.......'%torch.cuda.get_device_name(0))
    else:
      warnings.warn('CUDA not found! Execution may be slow......')
    

    # Load the video
    video_path = '/home/master_students/Atharva/SAM-ChatGPT_for_Kitchen_Tasks/Videos/cam_1_video.mp4' # Load the appropriate video path 
    video = cv2.VideoCapture(video_path)

    ## Flags
    yolo_all = False  #Toggle if you want to see all the detected objects or not

    output_path = '/home/master_students/Atharva/sam_centroid_cam_1.mp4'  # Load the appropriate video path
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
        
        if frame_counter >= 100:
            break
        result_frame, c_x, c_y = calculate_centroid(frame, YOLO, poi='apple', sam_centroid=True, yolo_all=yolo_all)
        print(f"Object Centroid: ({c_x}, {c_y})") if c_x != 0 else None
        cv2.imshow(result_frame) if c_x != 0 else None
        output_video.write(result_frame)
        frame_counter += 1

    print("Process Finished!!!")
    video.release()
    output_video.release()
    cv2.destroyAllWindows()

#    print("Starting file removal...")
#    try:
#    	os.remove("sam_vit_h_4b8939.pth")
#    	print("File removal complete!")
#
#    except FileNotFoundError:
#    	print("File not found.")
#    except Exception as e:
#    	print("An error occured while removing the file: ", e)
    
