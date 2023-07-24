import torch
import warnings
# import sys
# import torchvision
from torchvision.ops import nms
import cv2
# import matplotlib.pyplot as plt
import numpy as np
# import PIL
# import time
import math
import os, os.path
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
# import ultralytics
# from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
from itertools import chain
# import torchvision.transforms as transforms
# from classifier_model import Fruits_CNN


# Variables initializations
counter = 0
image_counter = 0
n_channels = 3

# pre_train_path = 'custom_nopretrain.pth'
# classifier = Fruits_CNN(num_channels=n_channels, num_classes=6)
# classifier.load_state_dict(torch.load(pre_train_path))
# classifier = classifier.cuda()
# classifier.eval()


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

	# Convert the mask to binary format (i.e., 0s and 1s only)
	binary_mask = mask_segmentation.astype(np.uint8)

	# Create index arrays representing the x and y coordinates of the pixels in the mask
	y, x = np.indices(binary_mask.shape)
	
	# Compute the x and y coordinats of the centroid by summing the product of the x-indices and the binary_mask values, then dividing by the total sum of the binary_mask values
	centroid_x = int(np.sum(x * binary_mask) / np.sum(binary_mask))
	centroid_y = int(np.sum(y * binary_mask) / np.sum(binary_mask))

	return centroid_x, centroid_y
  

def find_longest_line(contour, centroid):
	"""
	This function finds the longest line through the centroid within a contour. 
	The longest line is computed by finding the furthest point from the centroid in the contour and 
	drawing a line through the centroid and that point to the other side of the contour.

	Parameters:
	----------
	contour : ndarray
		The contour is a 2-dimensional numpy array representing the points in the contour. Each element in the 
		array should be a 2-dimensional point [x, y].
	
	centroid : list or ndarray
		The centroid is a list or a 1-dimensional numpy array representing the [x, y] coordinates of the centroid
		around which the longest line is to be calculated.

	Returns:
	-------
	tuple
		A tuple of two lists, each list representing the [x, y] coordinates of one of the two ends of the longest line.
		The first element of the tuple is the furthest point from the centroid on the contour.
		The second element of the tuple is the point on the other side of the contour.
	"""
	
	# Initializing variables
	max_distance = 0
	ind = 0

	# Iterate over all points in the contour
	for i in range(len(contour)):
		p1 = contour[i] # <- should be x1, y1
		# Calculate the Euclidean distance from the current point to the centroid and multiply by 2
		distance = 2 * (((p1[0] - centroid[0])**2 + (p1[1] - centroid[1])**2) ** .5)
		
		if distance > max_distance:  # Update the max_distanct if current distance is greater
			max_distance = distance
			ind = i
	# Get the x and y coordinates of the contour that has the max_distance
	x1 = contour[ind, 0]
	y1 = contour[ind, 1]

	# Calculate the x and y coordinates of the point on the other side of the contour, by reflecting the point across the centroid
	x2 = 2 * centroid[0] - x1
	y2 = 2 * centroid[1] - y1

	return [x1, y1], [x2, y2]


def get_rotation_angle(point1, point2, EVEN=False):
	"""
	This function calculates the angle by which the gripper needs to be rotated. 
	The calculation is based on two points that indicate the longest line and a boolean flag indicating the cut heuristic (EVEN).

	Parameters:
	----------
	point1 : list or ndarray
		This is a list or 1-dimensional numpy array representing the [x, y] coordinates of the first point on the longest line.
	
	point2 : list or ndarray
		This is a list or 1-dimensional numpy array representing the [x, y] coordinates of the second point on the longest line.
	
	EVEN : boolean, optional
		This flag indicates the cut heuristic. If it is set to True (even), the function calculates the angle to rotate the gripper to be perpendicular to the longest line. If it is set to False (odd), the function calculates the angle to rotate the gripper to be parallel to the longest line. The default is False.

	Returns:
	-------
	float
		The angle in degrees by which the gripper should be rotated. 
		The angle is calculated relative to the base vector [1, 0], and it's between 0 and 180 degrees. 
		If the angle can't be calculated (for example, when the direction vector is [0,0]), the function returns 0. 
		If EVEN is True, the function adds 90 degrees to the calculated angle.
	"""
	
	base_vec = np.array([1, 0])  # Define base vector
	dir_vec = np.array([point2[0] - point1[0], point2[1] - point1[1]])	# Compute direction vector from point1 to point2
	dir_vec = dir_vec / np.linalg.norm(dir_vec)	# Normalize the direction vector
	angle = math.degrees(np.arccos(np.clip(np.dot(dir_vec, base_vec), -1.0, 1.0)))	# Compute the angle between base vector and direction vector

	if math.isnan(angle):	# If the angle is not a number (can happen when dir_vec is [0,0]), set it to 0
		angle = 0

	if EVEN:  # If EVEN is true, we need to add 90 degrees to the angle
		angle+=90

	return angle


def get_blade_bb(c_x, c_y, rot_angle):
	"""
	This function calculates the pixel bounds of the projected blade in the scene. 
	The bounds are computed based on the given center coordinates (c_x, c_y), rotation angle (rot_angle), and a predefined blade half length.

	Parameters:
	----------
	c_x : float or int
		The x-coordinate of the blade's center in the scene.
	
	c_y : float or int
		The y-coordinate of the blade's center in the scene.
	
	rot_angle : float
		The angle in degrees at which the blade is rotated.

	Returns:
	-------
	list
		A list of two lists, each representing the [x, y] coordinates of one of the two corners of the bounding box around the projected blade. 
		The first element of the list represents the minimum [x, y], and the second element represents the maximum [x, y].
		minx, miny - the coordinates of the lower-left corner of the bounding box.
		maxx, maxy - the coordinates of the upper-right corner of the bounding box.

	Notes:
	------
	The function assumes a predefined blade half length (in pixels) which is used for the calculation of the bounding box.
	Adjust this length based on the actual dimensions of the blade in the scene.
	"""
	
	# Define half length of the blade in pixels. NOTE: this value is in pixels and needs to be tuned based on the actual blade dimensions.
	half_lenth = 120 

	# Calculate x and y distances using the cosine and sine of the rotation angle respectively
	x_dist = abs(half_lenth * np.cos(math.radians(rot_angle)))
	y_dist = abs(half_lenth * np.sin(math.radians(rot_angle)))

	print("\n\nx_dist: ", x_dist)
	print("\n\ny_dist: ", y_dist)

	minx = c_x - x_dist  # Calculate the minimum x coordinate of the bounding box
	maxx = c_x + x_dist  # Calculate the maximum x coordinate of the bounding box
	miny = c_y - y_dist  # Calculate the minimum y coordinate of the bounding box
	maxy = c_y + y_dist  # Calculate the maximum y coordinate of the bounding box 

	return [[minx, miny], [maxx, maxy]]


def check_collision(bbox, blade_bb):
	"""
	This function checks if the bounding box of a blade intersects with another bounding box in the scene, 
	indicating a collision. It returns True if there's a collision, and False otherwise.

	Parameters:
	----------
	bbox : list
		A list of two lists, each representing the [x, y] coordinates of one of the two corners of the bounding box.
		The first element of the list represents the minimum [x, y] (lower-left corner), and the second element represents 
		the maximum [x, y] (upper-right corner).

	blade_bb : list
		A similar list of two lists, each representing the [x, y] coordinates of one of the two corners of the blade's bounding box.

	Returns:
	-------
	boolean
		Returns True if the bounding boxes intersect (indicating a collision), and False otherwise.

	Notes:
	------
	This function uses the Axis-Aligned Bounding Box (AABB) method for collision detection.
	"""
	
	print("bb col check")
	
	# If the left edge of the blade is to the right of the right edge of the bbox, no collision
	if blade_bb[0][0] > bbox[1][0]:
		print(blade_bb[0][0])
		print(">",  bbox[1][0])
		return False
	
	# If the right edge of the blade is to the left of the left edge of the bbox, no collision
	elif blade_bb[1][0] < bbox[0][0]:
		print("\n ", blade_bb[1][0])
		print("<",  bbox[0][0])
		return False
	
	# If the bottom edge of the blade is above the top edge of the bbox, no collision
	elif blade_bb[0][1] > bbox[1][1]:
		print("\n ", blade_bb[0][1]) 
		print(">",  bbox[1][1])
		return False
	
	# If the top edge of the blade is below the bottom edge of the bbox, no collision
	elif blade_bb[1][1] < bbox[0][1]:
		print("\n ", blade_bb[1][1])
		print("<",  bbox[0][1])
		return False
	
	 # If none of the above conditions are met, the bounding boxes must intersect
	else:
		return True


def generate_SAM_centroid(image, anns, classifier, target, random_color=False, disp_centroid=False):
	"""
    This function finds the object of interest in an image, segments it, calculates its centroid, and overlays the segmentation on the original image.

    The function works as follows: 
    It iterates over each annotation in the list, creates a binary mask from the annotation, and segments the object in the image.
    Then, it uses a classifier to identify the segmented object. 
    If the identified object matches the target object, the function adds the mask and the area of the object to separate lists.
    If no objects match the target, the function selects the mask with the largest area.
    If more than one object matches the target, the function combines all the matching masks and sums their areas.
    Finally, the function calculates the centroid of the resulting mask, overlays the mask on the original image, and optionally displays the centroid.

    Parameters:
    ----------
    image: numpy.ndarray
        The original image.
    
    anns: list
        A list of annotations where each annotation is a dictionary containing 'area' and 'segmentation' as keys.
    
    classifier: Callable
        A classifier that takes an image as input and returns a list of detected objects in the image.
    
    target: str
        The name of the object to find in the image.
    
    random_color: bool, optional
        If True, the color of the mask will be random. If False, the color will be orange. Default is False.
    
    disp_centroid: bool, optional
        If True, the centroid of the mask will be displayed on the image. Default is False.

    Returns:
    -------
    tuple
        A 7-tuple containing:
        - output_img: numpy.ndarray. The image with the mask overlaid.
        - cent_x: int. The x-coordinate of the centroid.
        - cent_y: int. The y-coordinate of the centroid.
        - poi_area: float. The area of the segmented object of interest.
        - point1: list. The [x, y] coordinates of the first point on the longest line.
        - point2: list. The [x, y] coordinates of the second point on the longest line.
        - angle: float. The angle in degrees at which the longest line is oriented.
    """

	if len(anns) == 0: return	# If there are no masks, end the function
	
	print("Number of masks: ", len(anns))

	# Initializing variables
	global image_counter
	updated_masks_seg = []
	updated_masks_area = []

	# For each mask in the annotations, create a binary mask and create an image with the masked object
	for mask in anns:
		# Extract the segmentation and convert it to binary
		mask_seg = mask['segmentation']
		bin_mask = (mask_seg * 255).astype(np.uint8)

		inverted_mask = cv2.bitwise_not(bin_mask)	# Invert the mask

		# Copy the image and set the masked area to white
		interested_image = np.copy(image)
		interested_image[inverted_mask == 255] = 255

		# Add padding to the image
		top, bottom, left, right = [10]*4
		color_of_border = [255, 255, 255]	# Define the color of padding (white)
		padded_image = cv2.copyMakeBorder(interested_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color_of_border)

		# Save the masked image and the image with the masked object
		print(f"\nImage No: {image_counter}")
		print("Saving the mask image...")
		filename_masked = f'masked_image_{image_counter}.jpg'
		current_dir = os.getcwd()
		image_path_masked = os.path.join(current_dir, "Experiment Images", filename_masked)
		cv2.imwrite(image_path_masked, inverted_mask)
		print("Masked image saved at: ", image_path_masked)

		print("Saving the merged image...")
		filename_merged= f'merged_image_{image_counter}.jpg'
		current_dir = os.getcwd()
		image_path_merged = os.path.join(current_dir, "Experiment Images", filename_merged)
		cv2.imwrite(image_path_merged, padded_image)
		print("Merged image saved at: ", image_path_merged)
		image_counter += 1

		classifier_result = classifier(padded_image)	# Apply the classifier (YOLO) to the padded image

		# Find the class of the object in the mask and add the mask and its area to the list if it matches the target class
		classifier_detect = classifier_result[0]
		detect = classifier_detect.boxes.data.cpu().numpy()
		class_detect = detect[:, 5]
		classifier_names = np.array([classifier_detect.names[class_idx] for class_idx in class_detect])

		if len(classifier_names) != 0:
			string_pred = classifier_names[0]
		else:
			string_pred = ""

		# string_pred is the name of the label outputed by the classifer
		if string_pred == target:
			updated_masks_seg.append(mask_seg)
			updated_masks_area.append(mask['area'])

	# If no mask was found for the target class, use the mask with the largest area (False bounding box)
	if len(updated_masks_seg) == 0:
		poi = sorted(anns, key=lambda x: x['area'], reverse=True)
		updated_masks_seg = [poi[0]['segmentation']]
		updated_masks_area = [poi[0]['area']]

	# If there are multiple masks for the target class, combine them into a single mask and add up their areas
	elif len(updated_masks_seg) > 1:
		combined_mask = np.logical_or.reduce(updated_masks_seg)   # Combine all masks using logical OR operation
		updated_masks_seg = [combined_mask]   # Replace the list with the combined mask
		total_area = sum(updated_masks_area)	# Add up all the mask areas
		updated_masks_area = [total_area]	# Replace the list with the total area

	# Compute the centroid of the mask that contains the target
	poi_mask = updated_masks_seg[0]
	poi_area = updated_masks_area[0]
	cent_x, cent_y = get_centroid(poi_mask)
	
	# Convert the mask to uint8 and find contours
	poi_mask = (poi_mask * 255).astype(np.uint8)
	contours, _ = cv2.findContours(poi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	
	# Find the longest line and compute its angle
	cont = contours[0].reshape(contours[0].shape[0], 2)
	point1, point2 = find_longest_line(cont, [cent_x, cent_y])  #TODO: FIX IT LATER
	angle = get_rotation_angle(point1, point2)
	
	# Create a filled colored mask and apply it to the image
	color = (255, 165, 0) if not random_color else tuple(np.random.randint(0, 255, 3).tolist())
	mask_overlay = np.zeros(image.shape, dtype=np.uint8)	# Create an empty mask of zeros with the same shape as image
	cv2.drawContours(mask_overlay, contours, -1, color, thickness=cv2.FILLED)
	img = cv2.bitwise_and(image, mask_overlay)
	
	# If desired, display the centroid position on the image
	if disp_centroid:
		cv2.circle(img, (cent_x, cent_y), 5, (0, 255, 255), -1)  # Red color for centroid
		cv2.putText(img, f"({cent_x}, {cent_y})", (cent_x - 50, cent_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
	
	# Invert image and create mask from white pixels
	img_ = cv2.bitwise_not(img)
	_, mask = cv2.threshold(cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY), 254, 255, cv2.THRESH_BINARY_INV)
	mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
	
	# Combine the masked foreground and the background
	output_img = cv2.add(cv2.bitwise_and(img_, mask), cv2.bitwise_and(image, cv2.bitwise_not(mask)))

	return cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB), cent_x, cent_y, poi_area, point1, point2, angle


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

	results = model(image)
	result = results[0]

	if detect_all:
		img = result.plot()
		return img, []

	# If detect_all flag is not set, proceed with finding the target_class
	detections = results[0].boxes.data.cpu().numpy()
	detections =  detections[detections[:,-1].argsort()]  # <- Sorting detections according to their class index

	boxes = detections[:, :4]
	scores = detections[:, 4]
	classes = detections[:, 5]
	names = np.array([result.names[class_idx] for class_idx in classes])

	bbox_target_list = []
	for target_name in target_class:
		indices = np.where(names == target_name)[0]  # Search for target object

		
		if len(indices) != 0:  # Found a target object
			keep = nms(torch.from_numpy(boxes[indices]), torch.from_numpy(scores[indices]), iou_threshold=0.5)
			bbox_target_list.append(boxes[indices][keep.numpy()].astype(int).tolist())

		else:
			bbox_target_list.append([])

	return image, result.plot(), bbox_target_list

   

def calculate_centroid(frame, yolo_model, sam_model, poi='', yolo_centroid=False, sam_centroid=False, return_frame=True, display_mask=False, yolo_all=False):
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
		If True, displays the mask. Default is False.c[2], bc[3], display_mask)
			cent_list.append([centroid_x, centroid_y])
		
	yolo_all : bool, optional
		If True, YOLO is used for all objects. Default is False.

	Returns:
	-------
	tuple
		If return_frame is True, returns the frame with centroids and the centroids' coordinates.
		Otherwise, returns only the centroids' coordinates.
	"""

	centroid_x, centroid_y = 0, 0

	if yolo_all or poi == '':  # If you want to detect all objects within the frame
		result_frame = handle_yolo_all(frame, yolo_model, yolo_all, poi)
		return result_frame, []
	else:
		result_frame, yolo_img, box_coord = detect_objects(frame, yolo_model, target_class=poi)
	
	# Handle zero coordinates: cent_x, cent_y, poi_area,
	if len(box_coord) == 0:
		return handle_zero_coordinates(frame, return_frame)
	
	cent_list_per_item = []
	
	# Calculate centroid based on the method selected
	if yolo_centroid:
		cent_list = []
		for bc in box_coord:
			result_frame, centroid_x, centroid_y = calculate_yolo_centroid(frame, bc[0], bc[1], bc[2], bc[3])
			cent_list.append([centroid_x, centroid_y])
		return result_frame, cent_list if return_frame else cent_list
	
		
	elif sam_centroid:
		print("Box coordinate: ", box_coord)
		plan_frame = frame.copy()
		for i in range(len(box_coord)):
			cent_list = []
			for bc in box_coord[i]:
				result_frame, plan_frame, centroid_x, centroid_y, mask_area, lp_1, lp_2, angle = calculate_sam_centroid(frame, plan_frame, yolo_model, sam_model, poi[i], bc[0], bc[1], bc[2], bc[3], display_mask)
				if not (centroid_x == 0 and centroid_y == 0 and mask_area == 0):  # If no false bounding box
					cent_list.append([centroid_x, centroid_y, mask_area, bc[0], bc[1], bc[2], bc[3], angle])
			cent_list_per_item.append(cent_list)
	
		print("Checking for collisions...")
		collision_frames_list = []
		print("\nLen cent list: ", len(cent_list_per_item))
		print("len cent list [0]: ", len(cent_list_per_item[0]))
		print("Cent List: ", cent_list_per_item)
		for i in range(len(cent_list_per_item)):
			for j in range(len(cent_list_per_item[i])):
				print("\ni: ", i)
				print("j: ", j)
				# given centroid and angle generate blade bbox
				cent_x = cent_list_per_item[i][j][0]
				cent_y = cent_list_per_item[i][j][1]
				angle = cent_list_per_item[i][j][7]
				blade_bbox = get_blade_bb(cent_x, cent_y, angle)
				collisions = []

				col_frame = frame.copy()
				start_point = (int(blade_bbox[0][0]), int(blade_bbox[0][1]))
				end_point = (int(blade_bbox[1][0]), int(blade_bbox[1][1]))
				col_frame = cv2.rectangle(col_frame, start_point, end_point, color=(255, 0, 0), thickness=2)

				for k in range(len(cent_list_per_item)):
					for l in range(len(cent_list_per_item[k])):
						if k != i or l != j:
							obj_bbox = [[cent_list_per_item[k][l][3], cent_list_per_item[k][l][4]], [cent_list_per_item[k][l][5], cent_list_per_item[k][l][6]]]
							
							# check if bboxes intersect
							print("\nobj bbox: ", obj_bbox)
							print("Blade bbox: ", blade_bbox)
							if check_collision(obj_bbox, blade_bbox):
								print("\n====== Collision Detected ========")
								# get vector from centroid to collision centroid
								col_x = cent_list_per_item[k][l][0]
								col_y = cent_list_per_item[k][l][1]
								dir_vec = np.array([col_x - cent_x, col_y - cent_y])
								dir_vec = dir_vec / np.linalg.norm(dir_vec)
								base_vec = np.array([1, 0])

								# get angle of perpendicular vector
								print("Dot: ", np.dot(dir_vec, base_vec))
								print("Clip: ", np.clip(np.dot(dir_vec, base_vec), -1.0, 1.0))
								print("Arccos: ", np.arccos(np.clip(np.dot(dir_vec, base_vec), -1.0, 1.0)))
								push_angle = math.degrees(np.arccos(np.clip(np.dot(dir_vec, base_vec), -1.0, 1.0))) # + 90

								if math.isnan(push_angle):
									push_angle = 0

								# get point on bbox that is closest to obj centroid
								push_x = min(max(obj_bbox[0][0], cent_x), obj_bbox[1][0])
								push_y = min(max(obj_bbox[0][1], cent_y), obj_bbox[1][1])

								# append this point and angle to collisions list
								collisions.append([push_x, push_y, push_angle])

								col_frame = cv2.rectangle(col_frame, (int(obj_bbox[0][0]), int(obj_bbox[0][1])), (int(obj_bbox[1][0]), int(obj_bbox[1][1])), (0, 0, 255), 2)

				collision_frames_list.append(col_frame)

				# append collisions to cent_list[i] 
				cent_list_per_item[i][j].append(collisions)

		print("\n\nCent List: ", cent_list_per_item)

		return result_frame, yolo_img, plan_frame, collision_frames_list, cent_list_per_item if return_frame else cent_list_per_item




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

	if return_frame:
		return frame, []  
	else:
		return []



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

	yolo_centX, yolo_centY = (x2 + x1) // 2, (y2 + y1) // 2
	frame = draw_cross_centroid(frame, yolo_centX, yolo_centY, (0, 255, 0))
	return frame, int(yolo_centX), int(yolo_centY)



def calculate_sam_centroid(frame, plan_frame, YOLO, mask_generator, target, x1, y1, x2, y2, display_mask):
	"""
	This function calculates the centroid using SAM and draws it on the given frame.
	It also has an option to display the generated mask.

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

	global counter
	sam_centX = 0
	sam_centY = 0
	mask_area = 0

	cropped_image = frame[y1:y2, x1:x2]

	cropped_mask = mask_generator.generate(cropped_image)

	cropped_mask_img, cent_x, cent_y, mask_area, point1, point2, angle = generate_SAM_centroid(cropped_image, cropped_mask, YOLO, target)

	if display_mask and mask_area != 0:
		# print("Saving the cropped image...")
		filename_cropped = f'cropped_image_{counter}.jpg'
		# filename_bright = f'bright_image_{counter}.jpg' #TODO: DELETE LATER
		current_dir = os.getcwd()
		image_path_cropped = os.path.join(current_dir, "Experiment Images", filename_cropped)
		cv2.imwrite(image_path_cropped, cropped_image)
		# print("Cropped image saved at: ", image_path_cropped)
		# print("Saving the segmented cropped image...")
		filename_cropped_seg = f'cropped_seg_image_{counter}.jpg'
		current_dir = os.getcwd()
		image_path_cropped_seg = os.path.join(current_dir, "Experiment Images", filename_cropped_seg)
		cv2.imwrite(image_path_cropped_seg, cv2.cvtColor(cropped_mask_img, cv2.COLOR_RGB2BGR))
		# print("Segmented Cropped image saved at: ", image_path_cropped_seg)
		counter += 1
		frame[y1:y2, x1:x2] = cv2.cvtColor(cropped_mask_img, cv2.COLOR_RGB2BGR)

	if cent_x != 0 and cent_y != 0 and mask_area != 0:
		sam_centX, sam_centY = cent_x + x1, cent_y + y1

		point1 = [point1[0] + x1, point1[1] + y1]
		point2 = [point2[0] + x1, point2[1] + y1]

	print("\n-------------->Angle: ", angle)

	
	# frame = draw_circle_centroid(frame, point1[0], point1[1], mask_area, (255, 0, 0))
	# frame = draw_circle_centroid(frame, point2[0], point2[1], mask_area, (255, 0, 0))

	draw_longest_line(plan_frame, point1, point2, angle, (255, 0, 0))
	
	return frame, plan_frame, int(sam_centX), int(sam_centY), mask_area, point1, point2, angle



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
	
	line_length, thickness = 5, 2
	cv2.line(frame, (centX - line_length, centY - line_length), (centX + line_length, centY + line_length), color, thickness)
	cv2.line(frame, (centX + line_length, centY - line_length), (centX - line_length, centY + line_length), color, thickness)
	font, size, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2
	coordinates_text = f"({centX}, {centY})"
	cv2.putText(frame, coordinates_text, (centX - 50, centY - 50), font, size, color, thickness)
	return frame


def draw_longest_line(frame, pt1, pt2, angle, color):
	"""
	This function draws a line on the given frame (image), from pt1 to pt2. 
	It also annotates the frame with the angle of rotation.

	Parameters:
	----------
	frame : ndarray
		The input frame or image on which the line is to be drawn. It should be a 2-dimensional or 3-dimensional numpy array, typically read from an image or video source.
	
	pt1 : list or ndarray
		This is a list or 1-dimensional numpy array representing the [x, y] coordinates of the first point of the line.
	
	pt2 : list or ndarray
		This is a list or 1-dimensional numpy array representing the [x, y] coordinates of the second point of the line.
	
	angle : float
		The angle in degrees which will be printed as an annotation on the frame.
	
	color : tuple
		A tuple representing the color of the line and the text. For grayscale images, it could be a single integer. For colored images (e.g., RGB), it should be a tuple of three integers (e.g., (255, 0, 0) for blue).

	Returns:
	-------
	ndarray
		The frame or image with the line and the angle annotation drawn on it.
	"""
	
	font, size, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2

	# Draw the line on the frame
	cv2.line(frame, (pt1[0], pt1[1]), (pt2[0], pt2[1]), color, thickness)

	# Prepare the text for the angle annotation
	coordinates_text = f"({int(angle)})"

	# Put the text on the frame, offset a little bit from pt1
	cv2.putText(frame, coordinates_text, (pt1[0] - 50, pt1[1] - 10), font, size, color, thickness)

	return frame


def draw_circle_centroid(frame, centX, centY, area, color):
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
	
	cv2.circle(frame, (centX, centY), radius=5, color=color, thickness=cv2.FILLED)
	font, size, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2
	coordinates_text = f"({centX}, {centY}, {area})"
	cv2.putText(frame, coordinates_text, (centX - 50, centY - 10), font, size, color, thickness)
	return frame


# if __name__ == '__main__':

# 	#============= Checking for cuda =======================
# 	print("Checking for cuda...")
# 	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 	if device.type == 'cuda':
# 	  print('CUDA is found! Executing on %s.......'%torch.cuda.get_device_name(0))
# 	else:
# 	  warnings.warn('CUDA not found! Execution may be slow......')
	
	
# 	#============= Loading the SAM Model =======================
# #    url = 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth'
# #    filename = 'sam_vit_h_4b8939.pth'

# 	# Make sure the file is present in the 'Models/' folder
# 	model_path_SAM = os.path.join('Models', 'sam_vit_h_4b8939.pth')
# 	print(model_path_SAM)
	
# 	if os.path.isfile(model_path_SAM):
# 		print("SAM model file exists!")
# 		model_type = "default"

# 		sam = sam_model_registry[model_type](checkpoint=model_path_SAM)
# 		sam.to(device=device)


# 		#TODO: Change SAM parameters
# 		SAM = SamAutomaticMaskGenerator(sam)
# 		# SAM = SamAutomaticMaskGenerator(model=sam,
# 		#                                 # points_per_side=32,
# 		#                                 # pred_iou_thresh=0.86,
# 		#                                 # stability_score_thresh=0.92,
# 		#                                 # crop_n_layers=1,
# 		#                                 crop_n_points_downscale_factor=2,
# 		#                                 min_mask_region_area=100,  # Requires open-cv to run post-processing
# 		#                                 ) 
							
# 	else:
# 		warnings.warn("The file does not exits.")
		
# #    if os.path.isfile("sam_vit_h_4b8939.pth"):
# #        print("File already exists!")
# #
# #    else:
# #        urllib.request.urlretrieve(url, filename, reporthook=show_progress)
# #        print("\nDownload complete!")
		
# #    sam_checkpoint = "sam_vit_h_4b8939.pth"
	
# 	#============= Loading the YOLO Model =======================
# 	print("Check if YOLO properly installed:")
# 	ultralytics.checks()  # Check if YOLO is installed properly

# 	# directory = '/home/master_students/Atharva/SAM-ChatGPT_for_Kitchen_Tasks/Models'
# 	model_path_YOLO = os.path.join('Models', 'yolov8n.pt')
# 	print(model_path_YOLO)

# 	if os.path.isfile(model_path_YOLO):
# 		print("YOLO model file exists!")
# 		YOLO = YOLO(model_path_YOLO)  # load a pretrained YOLOv8n detection model
# 		YOLO.to(device=device)
# 	else:
# 		warnings.warn("The file does not exits.")
	

# 	# Load the video
# 	video_name = 'cam_1_og_video.mp4'
# 	video_path = os.path.join('Videos/Test Videos', video_name) # Load the appropriate video path 
# 	if os.path.isfile(video_path):
# 		print("Video file exists!")
# 		video = cv2.VideoCapture(video_path)
# 	else:
# 		warnings.warn("The file does not exits.")
	

# 	## Flags
# 	yolo_all = False  #Toggle if you want to see all the detected objects or not

# 	# Specifying output video file:
# 	output_path = os.path.join('Videos/Test Videos', 'sam_mask_live_centroid_video.mp4')  # Load the appropriate video path
# 	fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# 	output_video = cv2.VideoWriter(output_path, fourcc, 30.0, (int(video.get(3)), int(video.get(4))))

# 	# Get total number of frames
# 	num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

# 	frame_counter = 0
# 	while video.isOpened():
# 		ret, frame = video.read()
		
# 		# Break the loop if the video ends
# 		if not ret:
# 			break
# 		result_frame, centroid_list = calculate_centroid(frame, YOLO, SAM, poi='apple', sam_centroid=True, yolo_all=False, display_mask=True)
# 		# cv2.imshow(f'Centroid Detection', results)
# 		# if len(centroid_list) == 0:
# 		#     pass
# 		# else:
# 		#     print("Centroid List: " , centroid_list)
# #             cv2.imshow("Final frame", result_frame)  # Comment out this line if using SSH to run the code
# 		output_video.write(result_frame)
# 		#else:
# 			#output_video.write(results)
# 		#frame_counter += 1

# 	print("Process Finished!!!")
# 	print(f"Output video saved at: {output_path}")
# 	video.release()
# 	output_video.release()
# 	cv2.destroyAllWindows()