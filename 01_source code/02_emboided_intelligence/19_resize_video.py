"""
created by

"""
import os, time, sys
home_dir = os.environ['HOME']
sys.path.append(home_dir + '/motion_tracking_ws/devel/lib/python3/dist-packages')
sys.path.append(home_dir + '/ws_tracking/devel/lib/python3/dist-packages')
sys.path.append(home_dir + 'dataSSD/ws_ros/devel/lib/python3/dist-packages')
sys.path.append('/opt/ros/noetic/lib/python3/dist-packages')

# from IPython.display import display, Image, Audio
import cv2  # We're using OpenCV to read video, to install !pip install opencv-python
import base64
import datetime
import numpy as np
import yaml
import pickle
import matplotlib.pyplot as plt
from src.docs_proc import load_text_file, get_file_name, resize_clip_video


def extract_arrays_with_find(input_string):
    try:
        # Find the start and end indices of each array
        th_start = input_string.find("sig_th = [") + len("sig_th = [")
        th_end = input_string.find("]", th_start)

        ff_start = input_string.find("sig_ff = [") + len("sig_ff = [")
        ff_end = input_string.find("]", ff_start)

        mf_start = input_string.find("sig_mf = [") + len("sig_mf = [")
        mf_end = input_string.find("]", mf_start)

        # Extract and convert to lists
        sig_th = eval(f"[{input_string[th_start:th_end]}]")
        sig_ff = eval(f"[{input_string[ff_start:ff_end]}]")
        sig_mf = eval(f"[{input_string[mf_start:mf_end]}]")

        return sig_th, sig_ff, sig_mf
    except Exception as e:
        return f"An error occurred: {e}"


def resize_video(input_path, output_path, width, height):
    # Open the video file
    cap = cv2.VideoCapture(input_path)

    # Check if the video file was successfully opened
    if not cap.isOpened():
        print(f"Error: Unable to open video file {input_path}")
        return

    # Get the frames per second (fps) of the input video
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Get the codec of the input video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Create a VideoWriter object for the output video
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Process the video frame by frame
    while True:
        ret, frame = cap.read()

        if not ret:
            break  # End of video

        # Resize the frame
        resized_frame = cv2.resize(frame, (width, height))

        # Write the resized frame to the output video
        out.write(resized_frame)

    # Release resources
    cap.release()
    out.release()
    print(f"Video saved to {output_path}")


def main():
    dir_org_videos = home_dir + '/dataSSD/J_06_demos'
    flg_clip_duration = False
    '''
    new_width = 360
    new_height = 203
    
    new_width = 480
    new_height = 270

    new_width = 560
    new_height = 315
    flg_clip_duration = True
    
    new_width = 640
    new_height = 360
    '''
    new_width = 480
    new_height = 270
    flg_clip_duration = True
    max_duration = 10


    dir_new_videos = home_dir + '/dataSSD/J_06_demos/resize/' + str(new_width)
    if not os.path.exists(dir_new_videos):
        os.makedirs(dir_new_videos)

    names_org_video = get_file_name(dir_org_videos)
    for cnt_video, dir_video in enumerate(names_org_video):
        name_org_video = dir_video[len(dir_org_videos) + 1:-4]
        dir_output = dir_new_videos + '/' + name_org_video + '.mp4'
        if not os.path.exists(dir_output):
            resize_video(dir_video, dir_output, new_width, new_height)
            # resize_clip_video(dir_video, dir_output, new_width, new_height, flg_clip=flg_clip_duration, target_duration=max_duration)


    print("Done.")


if __name__ == '__main__':
    main()





