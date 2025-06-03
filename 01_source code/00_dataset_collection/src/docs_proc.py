#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author     :

import copy
import os, yaml
import math
import numpy as np
import pickle
import random
import argparse
import sys
import logging
import time
import datetime
import json
import shutil

"""
    This file defines some common functions for document processing
"""

def get_folder_name(file_dir_):
    """
    :param file_dir_: root dir of target documents  e.g.: home_dir + "/dataset/ycb_meshes_google/backup/003_typical"
    :return: all folder dirs, only for folders and ignore files
    """
    file_list = []
    for root, dirs, files in os.walk(file_dir_):
        if root.count('/') == file_dir_.count('/') + 1:
            file_list.append(root)
    file_list.sort()
    return file_list


def get_file_name(file_dir_):
    """
    :param file_dir_: root dir of target documents  e.g.: home_dir + "/dataset/ycb_meshes_google/backup/003_typical"
    :return: all files dirs, only for files and ignore folders
    """
    file_list = []
    for root, dirs, files in os.walk(file_dir_):
        # print(root)  # current path
        if root.count('/') == file_dir_.count('/'):
            for name in files:
                str = file_dir_ + '/' + name
                file_list.append(str)
    file_list.sort()
    return file_list


def extract_info(file_name):
    pos_name = file_name.find('name')
    str_end = file_name.find('.png')
    str_obj_info = file_name[pos_name:str_end]
    return str_obj_info


def delete_folder(folder_path):
    """
    delete a folder and all iterm inside
    :param folder_path: /path/to/your/folder
    :return:
    """
    # Check if the folder exists
    if os.path.exists(folder_path):
        # Use shutil.rmtree to delete the folder and all its contents
        shutil.rmtree(folder_path)
        print(f"Folder '{folder_path}' and all its contents have been deleted.")
    else:
        print(f"Folder '{folder_path}' does not exist.")


def read_docx(file_path):
    from docx import Document
    try:
        # Load the .docx document
        doc = Document(file_path)

        # Extract all text paragraphs
        text = []
        for paragraph in doc.paragraphs:
            text.append(paragraph.text)

        # Combine all paragraphs into a single string
        full_text = "\n".join(text)
        return full_text
    except Exception as e:
        return f"An error occurred: {e}"


def load_text_file(file_path):
    try:
        # Open the file in read mode
        with open(file_path, 'r', encoding='utf-8') as file:
            # Read the entire content as a single string
            text = file.read()
        return text
    except Exception as e:
        return f"An error occurred: {e}"


def resize_video(input_path, output_path, width, height):
    import cv2
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


if __name__ == '__main__':
    home_dir = os.environ['HOME']
    dir_folder = home_dir + "/dataSSD/delete_test"
    delete_folder(dir_folder)

    #
    print("Test done.")