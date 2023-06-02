import torch
import warnings
import sys
import torchvision
import cv2
import matplotlib.pyplot as plt
import numpy as np
import PIL
import time
import os
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
    print("HERE!")
    #cv2.imshow('image', image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    

    if len(anns) == 0: return
    
    # Compute the centroid of the largest mask
    poi_mask = sorted(anns, key=lambda x: x['area'], reverse=True)[0]['segmentation']
    cent_x, cent_y = get_centroid(poi_mask)
    
    print("Cx, Cy: {}, {}".format(cent_x, cent_y))
    
    poi_mask = (poi_mask * 255).astype(np.uint8)
    
    # Convert the mask to uint8 and find contours
    contours, _ = cv2.findContours(poi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print("Cx, Cy: {}, {}".format(cent_x, cent_y))
    
    # Create a filled colored mask and apply it to the image
    color = (255, 165, 0) if not random_color else tuple(np.random.randint(0, 255, 3).tolist())
    
    # Create an empty mask of zeros with the same shape as image
    mask_overlay = np.zeros(image.shape, dtype=np.uint8)
    
    cv2.drawContours(mask_overlay, contours, -1, color, thickness=cv2.FILLED)
    img = cv2.bitwise_and(image, mask_overlay)
    
    
    if disp_centroid:
        # Display centroid position on image
        cv2.circle(img, (cent_x, cent_y), 5, (0, 255, 255), -1)  # Red color for centroid
        cv2.putText(img, f"({cent_x}, {cent_y})", (cent_x - 50, cent_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
    
    # Invert image and create mask from white pixels
    img_ = cv2.bitwise_not(img)
    _, mask = cv2.threshold(cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY), 254, 255, cv2.THRESH_BINARY_INV)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    # Combine the masked foreground and the background
    output_img = cv2.add(cv2.bitwise_and(img_, mask), cv2.bitwise_and(image, cv2.bitwise_not(mask)))
    #cv2.imshow('output_img', output_img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
    print("Cx, Cy 3: {}, {}".format(cent_x, cent_y))
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
    
    boxes_list = []
    
    # If detect_all flag is set to True, plot all detected objects on the image and returns the image
    if detect_all:
        img = result.plot()
        return img, boxes_list

    boxes = result.boxes.cpu().numpy()  # Extract bounding box coordinates and convert them to numpy array
    names = np.array([result.names[i] for i in boxes.cls])  # Extract class names for detected objects

    # Find indices of the detected objects that match the target_class
    indices = np.where(names == target_class)  # Search for target object
    print("Indices: ", indices[0], len(indices[0]))
     
    # If target_class is found in the detected objects, extract its bounding box coordinates
    if len(indices[0]) != 0:  # Found a target object
    	for i in range(len(indices[0])):
    		box = boxes[indices[0][i]].xyxy[0].astype(int)  # Get the box coordinates of the target
    		boxes_list.append(box)
    	print("Box Coordinates: \n", boxes_list)
    	#x1, y1, x2, y2 = boxes[indices[0][0]].xyxy[0].astype(int)  # Get the box coordinates of the target
    	return image, boxes_list
    else:  # If target_class is not found, return the original image and empty list
    	return image, boxes_list

   

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
    centroid_x, centroid_y = 0, 0
    
    # Detect objects in frame
    
    if yolo_all or poi == '':  # If you want to detect all objects within the frame
        frame = handle_yolo_all(frame, model, yolo_all, poi)
        return frame
    else:
        frame, box_coord = detect_objects(frame, model, target_class=poi)

    # Handle zero coordinates
    if len(box_coord) == 0:
        return handle_zero_coordinates(frame, return_frame)
    
    cent_list = []
    # Calculate centroid based on the method selected
    if yolo_centroid:
    	for bc in box_coord:
    	    frame, centroid_x, centroid_y = calculate_yolo_centroid(frame, bc[0], bc[1], bc[2], bc[3])
    	    print("Centroid for {} cent x: {}, cent_y: {}".format(bc, centroid_x, centroid_y))
    	    cent_list.append([centroid_x, centroid_y])
    	print(cent_list)
    	return frame, cent_list if return_frame else cent_list
    	    
    elif sam_centroid:
        for bc in box_coord:
            frame, centroid_x, centroid_y = calculate_sam_centroid(frame, SAM, x1, y1, x2, y2, display_mask)
            cent_X_list.append(centroid_x), cent_Y_list.append(centrtoid_y)
        
        return frame, centroid_x, centroid_y if return_frame else centroid_x, centroid_y
            #cv2.imshow('frame 2: ({}, {})'.format(centroid_x, centroid_y), frame)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()  
    #cv2.imshow('frame 3: ({}, {})'.format(centroid_x, centroid_y), frame)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()



def handle_yolo_all(frame, model, yolo_all, poi):
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
        
    Returns:
    -------
    tuple
        The frame and coordinates of the detected object.
    """
    # This function handles yolo_all condition
    yolo_img, _ = detect_objects(frame, model, detect_all=yolo_all)
    if poi == '':
        warnings.warn("Warning! No point-of-interest object mentioned in 'poi' field. Returning bounding boxes of all items.")
    return yolo_img



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
    # This function handles the condition when there are no box coordinates
    if return_frame:
    	return frame, 0, 0  
    else:
    	return 0, 0



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
    This function calculates the centroid using SAM 
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
    start_time_sam = time.time()
    cropped_mask = mask_generator.generate(cropped_img)
    end_time_sam = time.time()
    print("Time elapsed SAM: {}s".format(np.abs(end_time_sam - start_time_sam)))
    cropped_mask_img, cent_x, cent_y = generate_SAM_centroid(cropped_img, cropped_mask)
    #cv2.imshow('output_img', cropped_mask_img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
    #print("Cx, Cy 4: {}, {}".format(cent_x, cent_y))
    if display_mask:
        frame[y1:y2, x1:x2] = cv2.cvtColor(cropped_mask_img, cv2.COLOR_RGB2BGR)
    sam_centX, sam_centY = cent_x + x1, cent_y + y1
    #print("SAM Cx, Cy: {}, {}".format(sam_centX, sam_centY))
    frame = draw_circle_centroid(frame, sam_centX, sam_centY, (0, 0, 255))
    
    #cv2.imshow('frame', frame)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
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
    font, size, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2
    coordinates_text = f"({centX}, {centY})"
    #print("HERE")
    #print(coordinates_text)
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
    #print("SAM Segmentation Centroid: x={}, y={}\n".format(centX, centY))
    cv2.circle(frame, (centX, centY), radius=5, color=color, thickness=cv2.FILLED)
    font, size, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2
    coordinates_text = f"({centX}, {centY})"
    cv2.putText(frame, coordinates_text, (centX - 50, centY - 10), font, size, color, thickness)
    return frame



if __name__ == '__main__':

    #============= Checking for cuda =======================
    print("Checking for cuda...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == 'cuda':
      print('CUDA is found! Executing on %s.......'%torch.cuda.get_device_name(0))
    else:
      warnings.warn('CUDA not found! Execution may be slow......')
    
    
    #============= Loading the SAM Model =======================
#    url = 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth'
#    filename = 'sam_vit_h_4b8939.pth'


    model_path_SAM = os.path.join('Models', 'sam_vit_b_01ec64.pth')
    print(model_path_SAM)
    
    if os.path.isfile(model_path_SAM):
        print("SAM model file exists!")
        model_type = "vit_b"

        sam = sam_model_registry[model_type](checkpoint=model_path_SAM)
        sam.to(device=device)

        SAM = SamAutomaticMaskGenerator(sam)
    else:
        warnings.warn("The file does not exits.")
        
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

    # directory = '/home/master_students/Atharva/SAM-ChatGPT_for_Kitchen_Tasks/Models'
    model_path_YOLO = os.path.join('Models', 'yolov8n.pt')
    print(model_path_YOLO)

    if os.path.isfile(model_path_YOLO):
        print("YOLO model file exists!")
        YOLO = YOLO(model_path_YOLO)  # load a pretrained YOLOv8n detection model
        YOLO.to(device=device)
    else:
        warnings.warn("The file does not exits.")
    

    # Load the video
    video_name = 'apple_cutting.mp4'
    video_path = os.path.join('Videos', video_name) # Load the appropriate video path 
    if os.path.isfile(video_path):
        print("Video file exists!")
        video = cv2.VideoCapture(video_path)
    else:
        warnings.warn("The file does not exits.")
    

    ## Flags
    yolo_all = True  #Toggle if you want to see all the detected objects or not

    output_path = os.path.join('Videos/Test Videos', 'yolo_all_' + video_name)  # Load the appropriate video path
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
        
        #if frame_counter >= 100:
        #    break
        results = calculate_centroid(frame, YOLO, poi='apple', yolo_centroid=True, yolo_all=yolo_all)
        if not yolo_all:
            print("Results[1]", results[1])
            if results[1] == 0:
                pass
            else:
                if len(results[1]) != 0 and (frame_counter <= 50 or len(results[1]) >= 2):
                    cv2.imshow("Final frame", results[0])
                    cv2.waitKey(500)
                    cv2.destroyAllWindows()
                #print(len(results[1]))    
            #print(f"Object Centroid: ({results[1]}, {results[2]})") if results[1] != 0 else None
            #if len(results[1]) != 0 and (frame_counter <= 50 or len(results[1]) >= 2):
        	    #cv2.imshow("Final frame", results[0])
        	    #cv2.waitKey(500)
        	    #cv2.destroyAllWindows()
            # print("Frame: ", frame_counter)
            output_video.write(results[0])
        else:
            output_video.write(results)
        frame_counter += 1

    print("Process Finished!!!")
    print(f"Output video saved at: {output_path}")
    video.release()
    output_video.release()
    cv2.destroyAllWindows()

  

