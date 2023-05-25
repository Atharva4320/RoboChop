import cv2
import numpy as np
import os
import time

# Edits the video according to your desire. Crops the video by zoom_factor, 
# trims the video from start to a specified end_duration. 
# The output_file name is name of the input_file from "input_file_loc": _______.mp4 
# and the save location of the output_video is same as the input_video unless otherwise mentioned as input parameter.
def video_editer(input_video_loc, zoom_factor=1, output_video_loc='', crop_video=False, 
                 disp_zoomed_frame=False, disp_freq=20, trim_video=False, trim_duration=10):
    # Open the video file
    cap = cv2.VideoCapture(input_video_loc)

    # Get the video's original width and height
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Get the frame rate of the video
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Calculate the width and height of the output
    new_width = int(width / zoom_factor)
    new_height = int(height / zoom_factor)

    # Calculate the difference for padding
    width_diff = (width - new_width) // 2
    height_diff = (height - new_height) // 2

    # Define the codec using VideoWriter_fourcc and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    if output_video_loc == '':
      # Split the path into directory and file
      file_location, input_file = os.path.split(input_video_loc)

      if crop_video:
          # Generate output video path
          output_file_name = "(x" + str(zoom_factor) + ")_" + input_file
          output_video_loc = file_location + '/' + output_file_name

      elif trim_video:
          # Generate output video path
          output_file_name = "(" + str(trim_duration) + "s)_trim_" + input_file
          output_video_loc = file_location + '/' + output_file_name

    out = cv2.VideoWriter(output_video_loc, fourcc, cap.get(cv2.CAP_PROP_FPS), (new_width, new_height))

    frame_counter = 0
    while(cap.isOpened()):
        ret, frame = cap.read()

        if ret:

            # Crop the frame
            frame = frame[height_diff:height_diff+new_height, width_diff:width_diff+new_width]
            
            # Calculate the elapsed time
            frame_counter += 1
            elapsed_time = frame_counter / fps
            minutes, seconds = divmod(elapsed_time, 60)

            if trim_video and (seconds >= trim_duration):
                break
                cap.release()
                out.release()
                cv2.destroyAllWindows()

                # print(f"TRIM HERE!!! Elapsed time: {seconds}s")
            if disp_zoomed_frame:
                if frame_counter % disp_freq:
                    print(f'Frame no.: {frame_counter}')
                    cv2_imshow(frame)
                    print(f'Frame displaying 1 per {disp_freq} frames.')

            out.write(frame)

        else:
            break

    # Print after exporting:
    if crop_video:
        print("Finished cropping "+ str(zoom_factor) + " times zoomed " + input_file + " video!")
    elif trim_video:
        print("Finished trimming the video to "+ str(trim_duration) + "s from start of " + input_file + " video!")
    
    print("Output loc:")
    print(output_video_loc)
    # Release everything after the job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    
if __name__ == '__main__':

    if 'google.colab' in str(get_ipython()):  # Checking if running on google colab
        from google.colab.patches import cv2_imshow
        from google.colab import drive
        drive.mount('/content/drive')

    video_editer('/content/drive/MyDrive/Fluid Segmentation Drive/(x3)_cam_1_video.mp4', trim_video=True)
