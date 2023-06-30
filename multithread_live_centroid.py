import os
import torch
import warnings
import cv2
import numpy as np
import pyrealsense2 as rs
from SAM_YOLO_Centroid_Detection import *
import ultralytics
from ultralytics import YOLO
import UdpComms as U
import time 
import matplotlib.pyplot as plt
import open3d as o3d
import queue
import threading

# Package versions:
print("Np version: ", np.__version__)
print("CV2 version: ", cv2.__version__)


# dictionary of camera serials
# 1: '220222066259',
# 2: '151322066099',
# 3: '151322069488',
# 4: '151322061880',
# 5: '151322066932'

# Camera:
#serial_number = '220222066259'  # Replace with the desired camera's serial number



# pub = Publisher(5502)  # Generate a publisher
 

#TODO: VISION LOOP
def vision_loop(img_queue, verts_queue, mask_queue, udp):
	W = 848
	H = 480

	# Camera 1: 
	serial_number_1 = '220222066259'
	pipeline_1 = rs.pipeline()
	config_1 = rs.config()
   
	pipeline_wrapper_1 = rs.pipeline_wrapper(pipeline_1)
	pipeline_profile_1 = config_1.resolve(pipeline_wrapper_1)
	device_1 = pipeline_profile_1.get_device()
	device_product_line_1 = str(device_1.get_info(rs.camera_info.product_line))
   
	config_1.enable_device(serial_number_1)
	config_1.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30)
	config_1.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)


	# align stream
	aligned_stream = rs.align(rs.stream.color)
	point_cloud = rs.pointcloud()

	# start streaming
	pipeline_1.start(config_1)

	##-------------------------------------------------------------------------------------------------------------
	# Camera 3:
	serial_number_3 = '151322069488'
	pipeline_3 = rs.pipeline()
	config_3 = rs.config()
   
	pipeline_wrapper_3 = rs.pipeline_wrapper(pipeline_3)
	pipeline_profile_3 = config_3.resolve(pipeline_wrapper_3)
	device_3 = pipeline_profile_3.get_device()
	device_product_line_3 = str(device_3.get_info(rs.camera_info.product_line))
   
	config_3.enable_device(serial_number_3)
	config_3.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30)
	config_3.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)


	# align stream
	aligned_stream = rs.align(rs.stream.color)
	point_cloud = rs.pointcloud()

	# start streaming
	pipeline_3.start(config_3)

	# Define the codec and create a VideoWriter object
	fourcc = cv2.VideoWriter_fourcc(*'mp4v')
	output_path_1 = os.path.join('Videos/Test Videos/SAM_refined', 'cam_1_video_SAM_refined_11.mp4')
	output_path_3 = os.path.join('Videos/Test Videos/SAM_refined', 'cam_3_video_SAM_refined_11.mp4')
	output_path_mask = os.path.join('Videos/Test Videos/SAM_refined', 'mask_video_SAM_refined_11.mp4')
	out_1 = cv2.VideoWriter(output_path_1, fourcc, 30, (W, H))
	out_3 = cv2.VideoWriter(output_path_3, fourcc, 30, (W, H))
	mask_out = cv2.VideoWriter(output_path_mask, fourcc, 1, (W, H))

	while True :
		# Get the frames of camera 1
		frames_1 = pipeline_1.wait_for_frames()
		frames_1 = aligned_stream.process(frames_1)
		color_frame_1 = frames_1.get_color_frame()
		color_image_1 = np.asanyarray(color_frame_1.get_data())
		depth_frame = frames_1.get_depth_frame().as_depth_frame()

		points = point_cloud.calculate(depth_frame)
		verts = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, W, 3)


		# Get the frames of camera 3
		frames_3 = pipeline_3.wait_for_frames()
		frames_3 = aligned_stream.process(frames_3)
		color_frame_3 = frames_3.get_color_frame()
		color_image_3 = np.asanyarray(color_frame_3.get_data())

		# Display the videos
		cv2.imshow("Camera 3", color_image_3)
		# cv2.imshow("Camera 1", color_image_1)

		# Save the cam 3 video:
		out_3.write(color_image_3)

		obs_message = udp.ReadReceivedData()
		# print(obs_message)
		if obs_message == "Segment":
			img_queue.put(color_image_1)
			print("img_queue: ", img_queue)
			verts_queue.put(verts)
		
		if not mask_queue.empty():
			mask = mask_queue.get()
			# display mask on color_image_1 as well as separate cv2.imshow window
			cv2.imshow("Mask Image", mask)
			mask_out.write(mask)
		else:
			out_1.write(color_image_1)

		# Press 'q' or 'esc' to break the loop
		if cv2.waitKey(1) & 0xFF == ord('q') or cv2.waitKey(1) == 27:
			# Release resources and close the window
			out_1.release()
			out_3.release()
			pipeline_1.stop()
			pipeline_3.stop()
			cv2.destroyAllWindows()
			break
			 

#TODO: SAM LOOP
def SAM_loop(img_queue, verts_queue, mask_queue, udp, YOLO, SAM):
	 
	 while True:
			if not img_queue.empty():
				color_image_1 = img_queue.get()
				verts = verts_queue.get()
					
				# SAM STUFF HERE
				print('\nSend the coordinates!!\n')

				result_frame, centroid_list = calculate_centroid(color_image_1, YOLO, SAM, poi='Apple', sam_centroid=True,display_mask=True)
				# print("SAM")

				# frame, _ = calculate_centroid(color_image_1, YOLO, SAM, poi='', yolo_centroid=True,yolo_all=True)

				# cv2.imshow("Frame", frame)
				
				## in a for loop:
				coord_list = []
				if len(centroid_list) == 0:
					# message = None 
					continue
				else: 
					prev_loc = []
					for centroids in centroid_list:
						
						# centroids[0] -> centroid_x
						# centroids[1] -> centroid_y
						# centroids[2] -> mask_area
						# centroids[3] -> bc[0] -> x1
						# centroids[4] -> bc[1] -> y1
						# centroids[5] -> bc[2] -> x2
						# centroids[6] -> bc[3] -> y2
						# centroids[7] -> lp_1 -> [x,y]
						# centroids[8] -> lp_2 -> [x,y]

						print("centroid: ", centroids[1], centroids[0], "area: ", centroids[2])
						obj_points = verts[int(centroids[1]-10) : int(centroids[1]+10), int(centroids[0]-10) : int(centroids[0]+10)].reshape(-1,3)
					
						zs = obj_points[:,2]
						z = np.median(zs)
						xs = obj_points[:,0]
						ys = obj_points[:,1]
						ys = np.delete(ys, np.where((zs < z - 1) | (zs > z + 1))) # take only y for close z to prevent including background <-  #TODO Try removing
						x_pos = np.median(xs)
						y_pos = np.median(ys)
						z_pos = z

						corner1 = verts[int(centroids[4]), int(centroids[3])].reshape(-1,3)
						corner2 = verts[int(centroids[6]), int(centroids[5])].reshape(-1,3)
						
						# print(centroids[7])

						lp1 = verts[int(centroids[7][1]), int(centroids[7][0])].reshape(-1,3)
						lp2 = verts[int(centroids[8][1]), int(centroids[8][0])].reshape(-1,3)

						# print("x_pos, y_pos", x_pos, y_pos)

		### FIX ->      ### THIS NEEDS TO BE DONE PROPER###
						y_pos = -y_pos
						x_pos = -x_pos

						median_point = np.array([x_pos, y_pos, z_pos, centroids[2], corner1[:, 0][0], corner1[:, 1][0], corner1[:, 2][0], corner2[:, 0][0], corner2[:, 1][0],
			       corner2[:, 2][0], lp1[:, 0][0], lp1[:, 1][0], lp1[:, 2][0], lp2[:, 0][0], lp2[:, 1][0], lp2[:, 2][0]])
						
						print("\nPoints to append: ", median_point)

						coord_list.append(median_point)
						prev_loc = coord_list

					message_good = True
					for entry in coord_list:
						print("Entry: ", entry)
						if np.any(np.isnan(entry)):
							message_good = False
							print("Got a 'nan' value!")
							break

					if message_good:
						message = coord_list

					else:
						continue
				
				print("Centroids : ", message)
				# print("Centroid list: ", centroid_list)
				udp.SendData(str(message))  # Send the message
				mask_queue.put(result_frame)


if __name__ == '__main__':
	img_queue = queue.Queue()

	message = []
	centroid_list = []
	
	udp = U.UdpComms(udpIP='172.26.69.200', sendIP='172.26.5.54', portTX=5500, portRX=5501, enableRX=True)

	## SETUP:
	#============= Checking for cuda =======================
	print("Checking for cuda...")
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	if device.type == 'cuda':
		print('CUDA is found! Executing on %s.......'%torch.cuda.get_device_name(0))
	else:
		warnings.warn('CUDA not found! Execution may be slow......')
	   
	#============= Loading the SAM Model =======================
	model_path_SAM = os.path.join('Models', 'sam_vit_h_4b8939.pth')
	print(model_path_SAM)
	
	if os.path.isfile(model_path_SAM):
		print("SAM model file exists!")
		model_type = "default"

		sam = sam_model_registry[model_type](checkpoint=model_path_SAM)
		sam.to(device=device)

		SAM = SamAutomaticMaskGenerator(sam)
	else:
		warnings.warn("The file does not exits.")
	
	#============= Loading the YOLO Model =======================
	model_path_YOLO = os.path.join('Models', 'best.pt')
	print(model_path_YOLO)

	if os.path.isfile(model_path_YOLO):
		print("YOLO model file exists!")
		ultralytics.checks()
		YOLO = YOLO(model_path_YOLO)  # load a pretrained YOLOv8n detection model
		YOLO.to(device=device)
	else:
		print("The file does not exits.")

	img_queue = queue.Queue()
	verts_queue = queue.Queue()
	mask_queue = queue.Queue()

	vision = threading.Thread(target=vision_loop, args=(img_queue, verts_queue, mask_queue, udp))
	mask_sam = threading.Thread(target=SAM_loop, args=(img_queue, verts_queue, mask_queue, udp, YOLO, SAM))

	vision.start()
	mask_sam.start()
	