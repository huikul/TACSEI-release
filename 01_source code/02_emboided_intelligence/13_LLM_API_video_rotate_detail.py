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
from src.docs_proc import load_text_file

from openai import OpenAI
dir_api_key = home_dir + '/.openai/tams_openai_key_01'
with open(dir_api_key, 'r') as file:
    api_key = file.read()
api_key = api_key[0:len(api_key)-1]
os.environ["OPENAI_API_KEY"] = api_key


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


def main():
    dir_video = home_dir + '/dataSSD/tamser_video/rotate_counterclock_05.mp4'
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
    # Display the frames
    # for img in base64Frames:
    #     display(Image(data=base64.b64decode(img.encode("utf-8"))))
    #     time.sleep(0.025)  # Delay for visualization
    #
    file_path = 'prompts/01_role_description.txt'
    doc_role = load_text_file(file_path)
    # print(doc_role)
    file_path = 'prompts/02_hardware_description.txt'
    doc_hardware = load_text_file(file_path)
    print(doc_hardware)
    file_path = 'prompts/07_task_video_pin.txt'
    doc_task = load_text_file(file_path)
    print(doc_task)
    file_path = 'prompts/10_identification_clockwise.txt'
    doc_clockwise = load_text_file(file_path)
    print(doc_clockwise)

    #
    PROMPT_MESSAGES = [
        {
            "role": "system",
            "content": 'You are a video analyzer.'
        },
        {
            "role": "user",
            "content": doc_clockwise
        },
        {   "role": "user",
            "content": [
                "These are frames from a video containing an object with a symbol 'A'. Based on the __previous definition and symbol A on the object__, please determine whether the __object with the symbol A__ is __rotating clockwise or counterclockwise__.",
                *map(lambda x: {"image": x, "resize": 1080}, base64Frames[0::40]),
            ],
        },
    ]

    params = {
        "model": "gpt-4o",
        "messages": PROMPT_MESSAGES,
        "max_tokens": 16384,
        "temperature": 0.01,
    }

    client = OpenAI()
    completion = client.chat.completions.create(**params)
    print(completion.choices[0].message.content)
    # return
    #
    PROMPT_MESSAGES = [
        {   "role": "system",
            "content": doc_role
        },
        {   "role": "system",
            "content": doc_hardware
        },
        {   "role": "user",
            "content": str(completion.choices[0].message.content) + doc_task,
        },
    ]
    params = {
        "model": "gpt-4o",
        "messages": PROMPT_MESSAGES,
        "max_tokens": 16384,
        "temperature": 0.01,
    }
    completion = client.chat.completions.create(**params)
    # print(completion.choices[0].message.content)
    ''''''
    python_lines = []
    recording = False
    for line in completion.choices[0].message.content.splitlines():
        if "```" in line:  # Start or end of the code block
            recording = not recording
            continue
        if recording or line.strip().startswith(("def", "import", "#", "class")):
            python_lines.append(line)

    script = "\n".join(python_lines)

    # Run the extracted script
    execution_context = {}
    try:
        exec(script, execution_context)
    except Exception as e:
        print(f"An error occurred while executing the script: {e}")

    # Access variables or functions defined in the script
    if 'sig_th' in execution_context:
        sig_th = execution_context['sig_th']
        # print(f"Generated Variable: {sig_th}")
    if 'sig_ff' in execution_context:
        sig_ff = execution_context['sig_ff']
        # print(f"Generated Variable: {sig_ff}")
    if 'sig_mf' in execution_context:
        sig_mf = execution_context['sig_mf']
        # print(f"Generated Variable: {sig_mf}")
    dir_new = ''

    ''''''
    dir_save_data = "data"
    # np.save('data/test.npy', sig_th)
    str_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    dir_new = dir_save_data + "/13_LLM_API_time{}".format(str(str_time))
    if not os.path.exists(dir_new):
        os.makedirs(dir_new)
    #
    sig_th = np.array(sig_th)
    sig_ff = np.array(sig_ff)
    sig_mf = np.array(sig_mf)
    file_name = dir_new + "/sig_th.npy"
    np.save(file_name, sig_th)
    file_name = dir_new + "/sig_fh.npy"
    np.save(file_name, sig_ff)
    file_name = dir_new + "/sig_mh.npy"
    np.save(file_name, sig_mf)
    file_name = dir_new + "/01_role_description.txt"
    with open(file_name, "w", encoding="utf-8") as file:
        file.write(doc_role)
    file_name = dir_new + "/02_hardware_description.txt"
    with open(file_name, "w", encoding="utf-8") as file:
        file.write(doc_hardware)
    file_name = dir_new + "/06_task_video_bottle_cap.txt"
    with open(file_name, "w", encoding="utf-8") as file:
        file.write(doc_task)

    #
    ''''''
    plt.plot(sig_th, label='sig_th')
    plt.plot(sig_ff - 0.01, label='sig_ff')
    plt.plot(sig_mf + 0.01, label='sig_mf')
    plt.title('')
    plt.xlabel('Control step')
    plt.ylabel('Air pressure')
    plt.legend()
    plt.grid(True)
    file_name = dir_new + "/plot_image.png"
    plt.savefig(file_name, format="png", dpi=300)  # dpi=300 for high resolution
    plt.show()

    #

    print("Done.")


if __name__ == '__main__':
    main()


