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
from src.docs_proc import load_text_file, get_file_name

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


def worker(model, processor, prompt, dir_video):
    messages = [
        {
            "role": "system",
            "content": 'You are a video analyzer.'
        },
        {
            "role": "system",
            "content": prompt,
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": dir_video,
                },
                {"type": "text",
                 "text": "Find the swing object held by the person, and select a reference part on the **upper body of the object**, then describe the movement."
                },

            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    print("START.")
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(output_text)
    #
    #
    messages = [
        {
            "role": "system",
            "content": 'You are a video analyzer.'
        },
        {
            "role": "system",
            "content": prompt,
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": dir_video,
                },
                {
                    "type": "text",
                    "text": output_text[0],
                },
                {
                    "type": "text",
                    "text": "Focusing on the **upper body of the object**. Tell me if the object is swinging **clockwise** or **counterclockwise**, and explain the reason."
                },
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    print("START.")
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(output_text)
    return output_text[0]


def main():
    # The default range for the number of visual tokens per image in the model is 4-16384. You can set min_pixels and max_pixels according to your needs, such as a token count range of 256-1280, to balance speed and memory usage.
    # min_pixels = 256*28*28
    # max_pixels = 1280*28*28
    # processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int4")

    dir_demostrations = home_dir + '/dataSSD/J_06_demos/release'
    dir_task = '/swing/counterclockwise'
    lst_videos = get_file_name(dir_demostrations + dir_task)

    file_path = 'prompts/13_knowledge_swing_clockwise.txt'
    doc_counterclockwise = load_text_file(file_path)
    # doc_counterclockwise = ''
    print(doc_counterclockwise)
    #
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int4",
        torch_dtype="auto", device_map="auto", max_memory={0: 2048*1024*1024, 'cpu': 2048*1024*1024}
    )

    dir_save_data = "data"
    str_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    dir_new = dir_save_data + "/24_LLM_qwen_swing"
    if not os.path.exists(dir_new):
        os.makedirs(dir_new)

    file_name = dir_new + "/result_time{}.txt".format(str(str_time))
    for cnt_video, dir_video in enumerate(lst_videos):
        name_video = dir_video[len(dir_demostrations) + 1:-4]
        if name_video.find('swing') > -1:
            print("############", name_video)
            result = worker(model=model, processor=processor, prompt=doc_counterclockwise, dir_video=dir_video)
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





