"""
created by
"""
import torch
import pickle


def read_pickle():
    dir = '/homeL/XXX/Desktop/StateEstimation/output/'
    filename = 'output.pickle'

    with open(file=dir + filename, mode='rb') as file:
        info_fingers = pickle.load(file)
        data_length = len(info_fingers)
        print(info_fingers['lst_timestamp'][19])


def test_gpu():
    print("PyTorch Version: ", torch.__version__)

    if torch.cuda.is_available():
        print("CUDA is available")
        print("Device name:", torch.cuda.get_device_name(0))
    else:
        print("CUDA is not available")


if __name__ == '__main__':
    read_pickle()
