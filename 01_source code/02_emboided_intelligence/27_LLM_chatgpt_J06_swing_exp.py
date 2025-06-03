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

from openai import OpenAI
dir_api_key = home_dir + '/.openai/tams_openai_key_01'
with open(dir_api_key, 'r') as file:
    api_key = file.read()
api_key = api_key[0:len(api_key)-1]
os.environ["OPENAI_API_KEY"] = api_key

from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

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

def worker(prompt, dir_video):
    video = cv2.VideoCapture(dir_video)
    base64Frames = []
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))

    video.release()
    print(len(base64Frames), "frames read.")

    PROMPT_MESSAGES = [
        {
            "role": "system",
            "content": 'You are a video analyzer.'
        },
        {
            "role": "system",
            "content": prompt
        },
        {   "role": "user",
            "content": [
                "Focusing on the **upper body of the object**. Tell me if the object is swinging **clockwise** or **counterclockwise**, and explain the reason.",
                *map(lambda x: {"image": x, "resize": 768}, base64Frames[0::1]),
            ],
        },
    ]

    params = {
        "model": "gpt-4o",
        "messages": PROMPT_MESSAGES,
        "max_tokens": 16384,
    }
    client = OpenAI()
    completion = client.chat.completions.create(**params)
    print(completion.choices[0].message.content)
    return completion.choices[0].message.content


def main():
    dir_demostrations = home_dir + '/dataSSD/J_06_demos/release'
    dir_task = '/swing/counterclockwise'
    lst_videos = get_file_name(dir_demostrations + dir_task)

    file_path = 'prompts/13_knowledge_swing_clockwise.txt'
    doc_counterclockwise = load_text_file(file_path)
    # doc_counterclockwise = ''
    print(doc_counterclockwise)
    #
    #
    dir_save_data = "data"
    str_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    dir_new = dir_save_data + "/27_LLM_chatgpt_swing"
    if not os.path.exists(dir_new):
        os.makedirs(dir_new)

    file_name = dir_new + "/result_time{}.txt".format(str(str_time))
    for cnt_video, dir_video in enumerate(lst_videos):
        name_video = dir_video[len(dir_demostrations) + 1:-4]
        if name_video.find('swing') > -1:
            print("############", name_video)
            result = worker(prompt=doc_counterclockwise, dir_video=dir_video)
            with open(file_name, "a", encoding="utf-8") as file:
                file.write(str(name_video) + '           ' + str(result) + '\r\n\r\n')

    '''
    file_path = 'prompts/01_role_description.txt'
    doc_role = load_text_file(file_path)
    # print(doc_role)
    file_path = 'prompts/02_hardware_description.txt'
    doc_hardware = load_text_file(file_path)
    print(doc_hardware)
    file_path = 'prompts/07_task_video_pin.txt'
    doc_task = load_text_file(file_path)
    print(doc_task)
    file_path = 'prompts/09_definition_clockwise.txt'
    doc_clockwise = load_text_file(file_path)
    print(doc_clockwise)
    '''
    print("Done.")


if __name__ == '__main__':
    main()





