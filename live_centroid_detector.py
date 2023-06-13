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
#from UDPComms import Publisher
#from threading import Thread
import time 

# Package versions:
print("Np version: ", np.__version__)
print("CV2 version: ", cv2.__version__)


# dictionary of camera serials
# 1: '220222066259',
# 2: '151322066099',
# 3: '151322069488',
# 4: '151322061880',
# 5: '151322066932'

# Camera 1 (end-effector)
serial_number = '151322061880'  # Replace with the desired camera's serial number

W = 848
H = 480

# pub = Publisher(5502)  # Generate a publisher
 

if __name__ == '__main__':

    #udp = U.UdpComms(udpIP='172.26.69.200', sendIP='172.26.5.54', portTX=5500, portRX=5501)

    pipeline = rs.pipeline()
    config = rs.config()
   
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))
   
    config.enable_device(serial_number)
    config.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)


    # align stream
    aligned_stream = rs.align(rs.stream.color)
    point_cloud = rs.pointcloud()

    # start streaming
    pipeline.start(config)


    # Create a window to display the video
    #cv2.namedWindow('Video Stream', cv2.WINDOW_AUTOSIZE)

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = os.path.join('Videos/Test Videos', 'yolo_live_centroid_video.mp4')
    output_path_og = os.path.join('Videos/Test Videos', 'cam_1_og_video.mp4')
    out = cv2.VideoWriter(output_path, fourcc, 30, (W, H))
    out_og = cv2.VideoWriter(output_path_og, fourcc, 30, (W, H))


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
    model_path_YOLO = os.path.join('Models', 'yolov8n.pt')
    print(model_path_YOLO)

    if os.path.isfile(model_path_YOLO):
        print("YOLO model file exists!")
        ultralytics.checks()
        YOLO = YOLO(model_path_YOLO)  # load a pretrained YOLOv8n detection model
        YOLO.to(device=device)
    else:
        print("The file does not exits.")

    ## EXECUTION:
    duration = 90 
    count = 0
    
    while True :
        # Get the frames
        frames = pipeline.wait_for_frames()
        frames = aligned_stream.process(frames)
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        depth_frame = frames.get_depth_frame().as_depth_frame()

        points = point_cloud.calculate(depth_frame)
        verts = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, W, 3)

        # Write the color frame to the video file
        out_og.write(color_image)

        result_frame, centroid_list = calculate_centroid(color_image, YOLO, SAM, poi='apple', yolo_centroid=True)
        
        
        ## in a for loop:
        coord_list = []
        if len(centroid_list) == 0:
            message = None #"No centroid Detected!"
        else: 
            for centroids in centroid_list:
                obj_points = verts[int(centroids[1]-10) : int(centroids[1]+10), int(centroids[0]-10) : int(centroids[0]+10)].reshape(-1,3)
                
                zs = obj_points[:,2]
                z = np.median(zs)
                xs = obj_points[:,0]
                ys = obj_points[:,1]
                ys = np.delete(ys, np.where((zs < z - 1) | (zs > z + 1))) # take only y for close z to prevent including background
                x_pos = np.median(xs)
                y_pos = np.median(ys)
                z_pos = z

                median_point = np.array([x_pos, y_pos, z_pos])

                coord_list.append(median_point)

            message = coord_list
        
        print("Centroids : ", message)
        #udp.SendData(str(message))
        #pub.send(message)  # Send the message

        cv2.imshow("Final frame", result_frame)
        out.write(result_frame)
        
        # Press 'q' or 'esc' to break the loop
        if cv2.waitKey(1) & 0xFF == ord('q') or cv2.waitKey(1) == 27:
            break
        count += 1
       
    # Release resources and close the window
    out.release()
    out_og.release()
    pipeline.stop()
    cv2.destroyAllWindows()
