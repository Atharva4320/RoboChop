import os
import torch
import warnings
import cv2
import numpy as np
import pyrealsense2 as rs
from SAM_YOLO_Centroid_Detection import calculate_centroid
from ultralytics import YOLO

# Package versions:
print("Np version: ", np.__version__)
print("CV2 version: ", cv2.__version__)

# Camera 1 (end-effector)
serial_number = '220222066259'  # Replace with the desired camera's serial number

W = 848
H = 480

pipeline = rs.pipeline()
config = rs.config()
config.enable_device(serial_number)
config.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30)
config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)

# start streaming
pipeline.start(config)

# align stream
aligned_stream = rs.align(rs.stream.color)

# Create a window to display the video
cv2.namedWindow('Video Stream', cv2.WINDOW_AUTOSIZE)

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_path = os.path.join(os.path.expanduser('~'), 'cam_2_video.mp4')
out = cv2.VideoWriter(output_path, fourcc, 30, (W, H))

if __name__ == '__main__':

## SETUP:
    #============= Checking for cuda =======================
    print("Checking for cuda...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == 'cuda':
      print('CUDA is found! Executing on %s.......'%torch.cuda.get_device_name(0))
    else:
      warnings.warn('CUDA not found! Execution may be slow......')
    
    #============= Loading the YOLO Model =======================
    model_path_YOLO = os.path.join('Models', 'yolov8n.pt')
    print(model_path_YOLO)

    if os.path.isfile(model_path_YOLO):
        print("YOLO model file exists!")
        YOLO = YOLO(model_path_YOLO)  # load a pretrained YOLOv8n detection model
        YOLO.to(device=device)
    else:
        print("The file does not exits.")

## EXECUTION:
    while True:
        # Get the frames
        frames = pipeline.wait_for_frames()
        frames = aligned_stream.process(frames)
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())

        # Write the color frame to the video file
        out.write(color_image)

        results = calculate_centroid(color_image, YOLO, poi='apple', yolo_all=True)
        print(f"Object Centroid: ({results[1]}, {results[2]})") if results[1] != 0 else None
        # if results[1] != 0 and frame_counter <= 50:
        #     cv2.imshow("Final frame: ({}, {})".format(results[1], results[2]), results[0])
        #     cv2.waitKey(500)
        #     cv2.destroyAllWindows()
        # print("Frame: ", frame_counter)
        # Display the color frame
        cv2.imshow(f'Centroid Detection @ ({results[1]}, {results[2]})', results[0])
        
        # Write the color frame to the video file
        out.write(results[0])
        
        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources and close the window
    out.release()
    pipeline.stop()
    cv2.destroyAllWindows()

