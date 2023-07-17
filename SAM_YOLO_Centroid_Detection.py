import torch
import warnings
import sys
import torchvision
from torchvision.ops import nms
import cv2
import matplotlib.pyplot as plt
import numpy as np
import PIL
import time
import math
import os, os.path
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import ultralytics
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
from itertools import chain
import torchvision.transforms as transforms
from classifier_model import Fruits_CNN


counter = 0
image_counter = 0
n_channels = 3

pre_train_path = 'custom_nopretrain.pth'
classifier = Fruits_CNN(num_channels=n_channels, num_classes=6)
classifier.load_state_dict(torch.load(pre_train_path))
classifier = classifier.cuda()
classifier.eval()

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
  

def find_longest_line(contour, centroid):
	# contour = np.array(contour)
	center = np.mean(contour, axis=0)  # Compute the center of the contour
	max_distance = 0
	point1 = None
	point2 = None
	ind = 0
	for i in range(len(contour)):
		p1 = contour[i] # <- should be x1, y1
		distance = 2 * (((p1[0] - centroid[0])**2 + (p1[1] - centroid[1])**2) ** .5)
		if distance > max_distance:
			max_distance = distance
			ind = i
	x1 = contour[ind, 0]
	y1 = contour[ind, 1]
	x2 = 2 * centroid[0] - x1
	y2 = 2 * centroid[1] - y1

	return [x1, y1], [x2, y2]

def get_rotation_angle(point1, point2, EVEN=False):
	"""
	Return the angle to rotate the gripper to be at given the two points indicating the longest
	line and the cut heuristic (even --> perpendicular, odd --> parallel).
	"""
	base_vec = np.array([1, 0])
	dir_vec = np.array([point2[0] - point1[0], point2[1] - point1[1]])
	dir_vec = dir_vec / np.linalg.norm(dir_vec)
	angle = math.degrees(np.arccos(np.clip(np.dot(dir_vec, base_vec), -1.0, 1.0)))
	if EVEN:
		angle+=90
	return angle

def get_blade_bb(c_x, c_y, rot_angle):
	"""
	Calculate the pixel bounds of the projected blade in the scene
	"""
	half_lenth = 120 # NOTE: this is in pixels and needs to be tuned
	x_dist = half_lenth * np.cos(math.radians(rot_angle))
	y_dist = half_lenth * np.sin(math.radians(rot_angle))
	minx = c_x - x_dist
	maxx = c_x + x_dist
	miny = c_y - y_dist
	maxy = c_y + y_dist 
	return [[minx, miny], [maxx, maxy]]

def check_collision(bbox, c_x, c_y, rot_angle):
	"""
	Check if the line is in collision with the bounding box, return True/False.
	"""
	blade_bb = get_blade_bb(c_x, c_y, rot_angle)

	if blade_bb[0][0] > bbox[1][0]:
		return False
	elif blade_bb[1][0] < bbox[0][0]:
		return False
	elif blade_bb[0][1] > bbox[1][1]:
		return False
	elif blade_bb[1][1] < bbox[0][1]:
		return False
	else:
		return True
	
	


		# slope_1 = (centroid[1] - p1[1]) / (centroid[0] - p1[0])
		# for j in range(len(contour)):
		# 	p2 = contour[j]
		# 	if p1 == p2:
		# 		continue
		# 	slope_2 = (centroid[1] - p1[1]) / (centroid[0] - p1[0])

		# get unit vector from p1 to center
		# find p2 (point on opposite side) in direction of unit vector
		# get dist




		# for j in range(i + 1, len(contour)):
		# 	p2 = contour[j]
		# 	line_distance = np.abs(np.cross(p2 - p1, center - p1)) / np.linalg.norm(p2 - p1)
		# 	if line_distance > max_distance:
		# 		max_distance = line_distance
		# 		point1 = p1 # [x,y]
		# 		point2 = p2 # [x,y]

	# print("Points: ", point1, point2)
	# return point1, point2


def generate_SAM_centroid(image, anns, classifier, target, random_color=False, disp_centroid=False):
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
	
	print("Number of masks: ", len(anns))

	global image_counter
	# saved_segmentations = []
	# poi_mask = []
	# poi_area = 0 
	updated_masks_seg = []
	updated_masks_area = []
	for mask in anns:
		# print("HERE!")
		mask_seg = mask['segmentation']
		bin_mask = (mask_seg * 255).astype(np.uint8)

		inverted_mask = cv2.bitwise_not(bin_mask)
		interested_image = np.copy(image)

		interested_image[inverted_mask == 255] = 255

		top, bottom, left, right = [10]*4
		
		# Define the color of padding (white)
		color_of_border = [255, 255, 255]

		padded_image = cv2.copyMakeBorder(interested_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color_of_border)

		# cv2.imshow("Mask Image", inverted_mask)
		# cv2.imshow("Merged Image", padded_image)
		# cv2.waitKey(2000)  # Wait for 2000 ms (2 seconds) then close the window
		# cv2.destroyWindow("Mask Image")
		# cv2.destroyWindow("Merged Image")
		print(f"\nImage No: {image_counter}")
		print("Saving the mask image...")
		filename_masked = f'masked_image_{image_counter}.jpg'
		current_dir = os.getcwd()
		image_path_masked = os.path.join(current_dir, filename_masked)
		cv2.imwrite(image_path_masked, inverted_mask)
		print("Masked image saved at: ", image_path_masked)

		print("Saving the merged image...")
		filename_merged= f'merged_image_{image_counter}.jpg'
		current_dir = os.getcwd()
		image_path_merged = os.path.join(current_dir, filename_merged)
		cv2.imwrite(image_path_merged, padded_image)
		print("Merged image saved at: ", image_path_merged)
		image_counter += 1
		
		#Run the classifier to classify the padded image -> returns label name ('string')
		#---------------------------------------------------------------------------------
		label_dict = {0: 'Apple',
              1: 'Carrot',
              2: 'Cucumber',
              3: 'Lime',
              4: 'Orange',
              5: 'Pineapple'}
		img_height = 100
		img_width = 100
		transform_data = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                     transforms.Resize((img_height, img_width), antialias=True)])
		images = transform_data(padded_image)
		images = torch.unsqueeze(images, dim=0)
		images = images.cuda()
		images = images.float()
		print("\nimages shape: ", images.size())
		classifier_result = classifier(images)
		_, class_predicted = torch.max(classifier_result, 1)
		class_predicted = class_predicted.cpu().detach().numpy()[0]
		string_pred = label_dict[class_predicted]
		print("Classifier prediction: ", string_pred)
		#---------------------------------------------------------------------------------
		
		# cropped_name is the name of the label outputed by the classifer
		if string_pred == target:
			updated_masks_seg.append(mask_seg)
			updated_masks_area.append(mask['area'])

	if len(updated_masks_seg) == 0:
		# False bounding box:
		return image, 0, 0, 0, [0, 0], [0, 0]

	elif len(updated_masks_seg) > 1:
		# Combine all masks using logical OR operation
		combined_mask = np.logical_or.reduce(updated_masks_seg)
		# Replace the list with the combined mask
		updated_masks_seg = [combined_mask]

		 # Add up all the mask areas
		total_area = sum(updated_masks_area)
		# Replace the list with the total area
		updated_masks_area = [total_area]

	
	# Compute the centroid of the mask that contains the target
	poi_mask = updated_masks_seg[0]
	poi_area = updated_masks_area[0]

	cent_x, cent_y = get_centroid(poi_mask)
	
	poi_mask = (poi_mask * 255).astype(np.uint8)
	
	# Convert the mask to uint8 and find contours
	contours, _ = cv2.findContours(poi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	
	cont = contours[0].reshape(contours[0].shape[0], 2)
	point1, point2 = find_longest_line(cont, [cent_x, cent_y])  #TODO: FIX IT LATER
	
	angle = get_rotation_angle(point1, point2)
	
	# cv2.imshow("Mask", poi_mask)
	# cv2.waitKey(1000)
	# print(len(contours))
	# print(cont)
	# print(cont.shape)
	# img_cp = image.copy()

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

	print("Actual YOLO result:")
	print(detections)


	print("\nNames: ", names)
	print("\nBoxes: ", boxes)
	print("\nScores: ", scores)

	box_to_keep = []
	# for i, box in enumerate(boxes):
	# 	print(f"(x1={box[0]}, y1={box[1]}, x2={box[2]}, y2={box[3]})")
	# 	cropped_image = image[int(box[1] - 5):int(box[3] + 5), int(box[0] - 5):int(box[2] + 5)]
	# 	# Define the padding
	# 	top, bottom, left, right = [5]*4
	# 	# Define the color of padding (white)
	# 	color_of_border = [255, 255, 255]

	# 	cropped_image = cv2.copyMakeBorder(cropped_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color_of_border)
	# 	cropped_result = model(cropped_image)[0]
	# 	cropped_det = cropped_result.boxes.data.cpu().numpy()
	# 	print("\nCropped detection:")
	# 	print(cropped_det)
	# 	if len(cropped_det) != 0:

	# 		cropped_scores = cropped_det[0][4]
	# 		cropped_name = cropped_result.names[cropped_det[0][5]]
	# 		print(cropped_name, cropped_scores)




		# print("Actual name: ", np.array([cropped_result.names[class_ind] for class_ind in cropped_det[:, 5]]))




	# TODO: 
	# 	-> Check if the detection boxes are arranged in an order
	#		-> If so, check if the bounding boxes of the same classes are overlapping
	#	-> If detection not in order:
	#		*-> Sort the classes in a order [0, 1, 2, ...] <= {0=APPLE, 1=BANANA, ...}
	#		-> Run a loop where we select the bounding boxes that belong to one class
	#		-> Get their indices and check the overlap between them
	#		-> Store the bounding boxes per class


	#TODO: Multiple target objects
	bbox_target_list = []
	for target_name in target_class:
		indices = np.where(names == target_name)[0]  # Search for target object

		# print("Indices: ", indices, target_class)
		# print(len(indices))

		# print("ALL")
		# print("Boxes: ", boxes, type(boxes))
		# print("Scores: ", scores, type(scores))
		# print("Classes: ", classes, type(classes))
		# print("Names: ", names, type(names))
		# print(target_name, indices)
		if len(indices) != 0:  # Found a target object
			
			# print("TARGET")
			# print("Boxes: ", boxes[indices], type(boxes))
			# print("Scores: ", scores[indices], type(scores))
			# print("Classes: ", classes[indices], type(classes))
			# print("Names: ", names[indices], type(names))

			keep = nms(torch.from_numpy(boxes[indices]), torch.from_numpy(scores[indices]), iou_threshold=0.5)
			bbox_target_list.append(boxes[keep.numpy()].astype(int).tolist())
			# print("Keep: ", keep.numpy())
			# print("Boxes to keep:", boxes[keep.numpy()].astype(int).tolist())
		else:
			bbox_target_list.append([])
	# print("In detect_objects() function:")
	# print("Returned bounding boxes: ", bbox_target_list)
	return image, bbox_target_list

   

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
		# print("Classes to detect: ", len(poi))
		result_frame, box_coord = detect_objects(frame, yolo_model, target_class=poi)
	
	# print("\nBBOX COORDINATES: ", box_coord)

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
		for i in range(len(box_coord)):
			cent_list = []
			for bc in box_coord[i]:
				result_frame, centroid_x, centroid_y, mask_area, lp_1, lp_2, angle = calculate_sam_centroid(frame, yolo_model, sam_model, poi[i], bc[0], bc[1], bc[2], bc[3], display_mask)
				if not (centroid_x == 0 and centroid_y == 0 and mask_area == 0):  # If no false bounding box
					# cent_list.append([centroid_x, centroid_y, mask_area, bc[0], bc[1], bc[2], bc[3], lp_1, lp_2]) # TODO: OLDER VERSION
					cent_list.append([centroid_x, centroid_y, mask_area, bc[0], bc[1], bc[2], bc[3], angle])
			cent_list_per_item.append(cent_list)

	
	# ------- ADDED COLLISION DETECTION ---------
	# iterate through bounding boxes and get idx i
	for i in range(len(box_coord)):
		for j in range(len(box_coord[i])):
			# given centroid and angle generate blade bbox
			print(cent_list_per_item[i][j])
			cent_x = cent_list_per_item[i][j][0]
			cent_y = cent_list_per_item[i][j][1]
			angle = cent_list_per_item[i][j][7]
			collisions = []

			for k in range(len(box_coord)):
				for l in range(len(box_coord[k])):
					if k != i or l != j:
						obj_bbox = [[cent_list_per_item[k][l][3], cent_list_per_item[k][l][4]], [cent_list_per_item[k][l][5], cent_list_per_item[k][l][6]]]
						# check if bboxes intersect
						if check_collision(obj_bbox, cent_x, cent_y, angle):
							print("\n====== Collision Detected ========")
							# get vector from centroid to collision centroid
							col_x = cent_list_per_item[k][l][0]
							col_y = cent_list_per_item[k][l][1]
							dir_vec = np.array([col_x - cent_x, col_y - cent_y])
							dir_vec = dir_vec / np.linalg.norm(dir_vec)

							# get angle of perpendicular vector
							push_angle = math.degrees(np.arccos(np.clip(np.dot(dir_vec, dir_vec), -1.0, 1.0))) + 90

							# TODO: get point on SAM mask closest to obj centroid
							# get point on bbox that is closest to obj centroid
							push_x = min(max(obj_bbox[0][0], cent_x), obj_bbox[1][0])
							push_y = min(max(obj_bbox[0][1], cent_y), obj_bbox[1][1])

							# append this point and angle to collisions list
							collisions.append([push_x, push_y, push_angle])
			# append collisions to cent_list[i] 
			cent_list_per_item[i][j].append(collisions)

		print("\n\nCent List: ", cent_list_per_item)

		return result_frame, cent_list_per_item if return_frame else cent_list_per_item




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
	# This function calculates the centroid using yolo bounding box
	yolo_centX, yolo_centY = (x2 + x1) // 2, (y2 + y1) // 2
	frame = draw_cross_centroid(frame, yolo_centX, yolo_centY, (0, 255, 0))
	return frame, int(yolo_centX), int(yolo_centY)



def calculate_sam_centroid(frame, YOLO, mask_generator, target, x1, y1, x2, y2, display_mask):
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

	global counter
	# cv2.imshow("FRAME", frame)
	cropped_image = frame[y1:y2, x1:x2]

	# cv2.imshow("Cropped Image", cropped_image)
	# cv2.waitKey(5000)  # Wait for 2000 ms (2 seconds) then close the window
	# cv2.destroyWindow("Cropped Image")

	cropped_mask = mask_generator.generate(cropped_image)

		
	# cv2.destroyWindow("Cropped Image")
	
			
		

	'''
	View masks:
	- for loop:
		- convert binary mask where 1 -> 255, 0 -> 0  => (bin_mask * 255).astype(np.uint8)
		- visualize:
			cv2.imshow("Mask Image", mask)
			cv2.waitKey(5000)  # Wait for 2000 ms (2 seconds) then close the window
			cv2.destroyWindow("Mask Image")

	'''
	

	cropped_mask_img, cent_x, cent_y, mask_area, point1, point2, angle = generate_SAM_centroid(cropped_image, cropped_mask, classifier, target)


	if display_mask and mask_area != 0:
		print("Saving the cropped image...")
		filename_cropped = f'cropped_image_{counter}.jpg'
		current_dir = os.getcwd()
		image_path_cropped = os.path.join(current_dir, filename_cropped)
		cv2.imwrite(image_path_cropped, cropped_image)
		print("Cropped image saved at: ", image_path_cropped)
		print("Saving the segmented cropped image...")
		filename_cropped_seg = f'cropped_seg_image_{counter}.jpg'
		current_dir = os.getcwd()
		image_path_cropped_seg = os.path.join(current_dir, filename_cropped_seg)
		cv2.imwrite(image_path_cropped_seg, cv2.cvtColor(cropped_mask_img, cv2.COLOR_RGB2BGR))
		print("Segmented Cropped image saved at: ", image_path_cropped_seg)
		counter += 1
		frame[y1:y2, x1:x2] = cv2.cvtColor(cropped_mask_img, cv2.COLOR_RGB2BGR)

	if cent_x != 0 and cent_y != 0 and mask_area != 0:
		sam_centX, sam_centY = cent_x + x1, cent_y + y1

		point1 = [point1[0] + x1, point1[1] + y1]
		point2 = [point2[0] + x1, point2[1] + y1]

	print("\n-------------->Angle: ", angle)

	
	# frame = draw_circle_centroid(frame, point1[0], point1[1], mask_area, (255, 0, 0))
	# frame = draw_circle_centroid(frame, point2[0], point2[1], mask_area, (255, 0, 0))

	draw_longest_line(frame, point1, point2, angle, (255, 0, 0))
	
	return frame, int(sam_centX), int(sam_centY), mask_area, point1, point2, angle



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
	cv2.putText(frame, coordinates_text, (centX - 50, centY - 50), font, size, color, thickness)
	return frame


def draw_longest_line(frame, pt1, pt2, angle, color):
	font, size, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2
	cv2.line(frame, (pt1[0], pt1[1]), (pt2[0], pt2[1]), color, thickness)
	coordinates_text = f"({angle})"
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
	# This function draws circle centroid on the frame
	cv2.circle(frame, (centX, centY), radius=5, color=color, thickness=cv2.FILLED)
	font, size, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2
	coordinates_text = f"({centX}, {centY}, {area})"
	cv2.putText(frame, coordinates_text, (centX - 50, centY - 10), font, size, color, thickness)
	return frame

#TODO: Function defination
# def draw_longest_line(img, pt1, pt2, color, thickness=10, gap=1):
#     cv2.line(img, pt1, )
#     return img


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

	# Make sure the file is present in the 'Models/' folder
	model_path_SAM = os.path.join('Models', 'sam_vit_h_4b8939.pth')
	print(model_path_SAM)
	
	if os.path.isfile(model_path_SAM):
		print("SAM model file exists!")
		model_type = "default"

		sam = sam_model_registry[model_type](checkpoint=model_path_SAM)
		sam.to(device=device)


		#TODO: Change SAM parameters
		SAM = SamAutomaticMaskGenerator(sam)
		# SAM = SamAutomaticMaskGenerator(model=sam,
		#                                 # points_per_side=32,
		#                                 # pred_iou_thresh=0.86,
		#                                 # stability_score_thresh=0.92,
		#                                 # crop_n_layers=1,
		#                                 crop_n_points_downscale_factor=2,
		#                                 min_mask_region_area=100,  # Requires open-cv to run post-processing
		#                                 ) 
							
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
	video_name = 'cam_1_og_video.mp4'
	video_path = os.path.join('Videos/Test Videos', video_name) # Load the appropriate video path 
	if os.path.isfile(video_path):
		print("Video file exists!")
		video = cv2.VideoCapture(video_path)
	else:
		warnings.warn("The file does not exits.")
	

	## Flags
	yolo_all = False  #Toggle if you want to see all the detected objects or not

	# Specifying output video file:
	output_path = os.path.join('Videos/Test Videos', 'sam_mask_live_centroid_video.mp4')  # Load the appropriate video path
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
		result_frame, centroid_list = calculate_centroid(frame, YOLO, SAM, poi='apple', sam_centroid=True, yolo_all=False, display_mask=True)
		# cv2.imshow(f'Centroid Detection', results)
		# if len(centroid_list) == 0:
		#     pass
		# else:
		#     print("Centroid List: " , centroid_list)
#             cv2.imshow("Final frame", result_frame)  # Comment out this line if using SSH to run the code
		output_video.write(result_frame)
		#else:
			#output_video.write(results)
		#frame_counter += 1

	print("Process Finished!!!")
	print(f"Output video saved at: {output_path}")
	video.release()
	output_video.release()
	cv2.destroyAllWindows()