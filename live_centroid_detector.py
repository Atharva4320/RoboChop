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

W = 848
H = 480

# pub = Publisher(5502)  # Generate a publisher
 

if __name__ == '__main__':
    message = []
    centroid_list = []
    #%%  Cameras to test: 1 and 3  %%#
    ##-------------------------------------------------------------------------------------------------------------
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


    udp = U.UdpComms(udpIP='172.26.69.200', sendIP='172.26.5.54', portTX=5500, portRX=5501, enableRX=True)

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path_1 = os.path.join('Videos/Test Videos', 'cam_1_video.mp4')
    output_path_3 = os.path.join('Videos/Test Videos', 'cam_3_video.mp4')
    out_1 = cv2.VideoWriter(output_path_1, fourcc, 30, (W, H))
    out_3 = cv2.VideoWriter(output_path_3, fourcc, 30, (W, H))


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

    count = 0
    while True :
        # init_message = udp.ReadReceivedData()
        # if init_message is not None:
        #     print(init_message.split(','))
        # elif count % 1000000 == 0:
        #     print("No message")
        # # else:
        # #     continue
        # count += 1
            
        # Get the frames of camera 1
        frames_1 = pipeline_1.wait_for_frames()
        frames_1 = aligned_stream.process(frames_1)
        color_frame_1 = frames_1.get_color_frame()
        color_image_1 = np.asanyarray(color_frame_1.get_data())
        # color_image_1 = cv2.flip(color_image_1, 0)
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

        # Save the cam 3 video:
        out_3.write(color_image_3)

        # frame, centroid_list = calculate_centroid(color_image_1, YOLO, SAM, poi='', yolo_centroid=True,yolo_all=True)
        
        # frame_2, centroid_list = calculate_centroid(color_image_1, YOLO, SAM, poi='Apple', yolo_centroid=True, display_mask=False)
        
        # # out_1.write(color_image_1)
        # cv2.imshow("Frame", frame)
        # cv2.imshow("YOLO frame", frame_2)
    
        # # Press 'q' or 'esc' to break the loop
        # if cv2.waitKey(1) & 0xFF == ord('q') or cv2.waitKey(1) == 27:
        #     break

        obs_message = udp.ReadReceivedData()
        if obs_message == "Segment":
            print('\nSend the coordinates!!\n')
            result_frame, centroid_list = calculate_centroid(color_image_1, YOLO, SAM, poi='Apple', sam_centroid=True,display_mask=True)
            print("SAM")

            frame, _ = calculate_centroid(color_image_1, YOLO, SAM, poi='', yolo_centroid=True,yolo_all=True)

            cv2.imshow("Frame", frame)
        
            ## in a for loop:
            coord_list = []
            if len(centroid_list) == 0:
                # message = None 
                continue
            else: 
                prev_loc = []
                for centroids in centroid_list:
                    
                    print("centroid: ", centroids[1], centroids[0], "area: ", centroids[2])
                    obj_points = verts[int(centroids[1]-10) : int(centroids[1]+10), int(centroids[0]-10) : int(centroids[0]+10)].reshape(-1,3)
                    # all_points = verts[:, :].reshape(-1, 3)
                    # print('all_points', all_points.shape)
                    # mean_point = np.mean(all_points, axis=0)
                    # all_points_cropped = []
                    # print("mean_point", mean_point)
                    # for point in all_points:
                    #     if np.linalg.norm(point - mean_point) < 1:
                    #         all_points_cropped.append(point)
                    
                    # all_points = np.array(all_points_cropped)
                    # print('croped_all_points', all_points.shape)
                    # print("obj_points", obj_points)
                    # fig = plt.figure()
                    # ax = fig.add_subplot(projection='3d')
                    # ax.scatter(all_points[:,0], all_points[:,1], all_points[:,2], c='green')
                    # ax.scatter(obj_points[:,0], obj_points[:,1], obj_points[:,2], c='red')
                    # plt.show()

                    # obj = o3d.geometry.PointCloud()
                    # obj.points = o3d.utility.Vector3dVector(obj_points)
                    # pcl_colors = np.tile(np.array([1, 0, 0]), (len(obj_points),1))
                    # obj.colors = o3d.utility.Vector3dVector(pcl_colors)

                    # all = o3d.geometry.PointCloud()
                    # all.points = o3d.utility.Vector3dVector(all_points)
                    # all_colors = np.tile(np.array([0, 0, 1]), (len(all_points),1))
                    # all.colors = o3d.utility.Vector3dVector(all_colors)
                    # o3d.visualization.draw_geometries([all, obj])
                    
                    zs = obj_points[:,2]
                    z = np.median(zs)
                    xs = obj_points[:,0]
                    ys = obj_points[:,1]
                    ys = np.delete(ys, np.where((zs < z - 1) | (zs > z + 1))) # take only y for close z to prevent including background <-  #TODO Try removing
                    x_pos = np.median(xs)
                    y_pos = np.median(ys)
                    z_pos = z

                    # print("x_pos, y_pos", x_pos, y_pos)

    ### FIX ->      ### THIS NEEDS TO BE DONE PROPER###
                    y_pos = -y_pos
                    x_pos = -x_pos

                    median_point = np.array([x_pos, y_pos, z_pos, centroids[2]])

                    coord_list.append(median_point)
                    prev_loc = coord_list

                message_good = True
                for entry in coord_list:
                    if np.any(np.isnan(entry)):
                        message_good = False
                        break

                if message_good:
                    message = coord_list

                else:
                    continue
            
            print("Centroids : ", message)
            # print("Centroid list: ", centroid_list)
            udp.SendData(str(message))  # Send the message

            
        
        if obs_message == "Segment":
            for i in range(10):
                out_1.write(result_frame)
            cv2.imshow("Final frame", result_frame)
            cv2.waitKey(1000)

        # Press 'q' or 'esc' to break the loop
        if cv2.waitKey(1) & 0xFF == ord('q') or cv2.waitKey(1) == 27:
            break

        else:
            out_1.write(color_image_1)
            cv2.imshow("Final frame", color_image_1)
            continue
       
    # Release resources and close the window
    out_1.release()
    out_3.release()
    pipeline_1.stop()
    pipeline_3.stop()
    cv2.destroyAllWindows()
