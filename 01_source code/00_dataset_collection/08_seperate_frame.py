"""
created by

"""
import os, time, sys
home_dir = os.environ['HOME']

import datetime
import numpy as np
import yaml

import pickle
import random
from src.docs_proc import get_file_name, get_folder_name, delete_folder


def worker(dir_dataset, dir_save_new, cfg_info, root_dir_dataset):
    '''
    if cfg_info['dataset_separate']['dir_save_new'] == "None":
        dir_save_new = dir_dataset
    else:
        dir_save_new = cfg_info['dataset_separate']['dir_save_new']
        # delete the previous data if necessary
        if os.path.exists(dir_save_new) and cfg_info['dataset_separate']['flg_delete_previous_data'] is True:
            delete_folder(dir_save_new)
        if not os.path.exists(dir_save_new):
            os.makedirs(dir_save_new)
    '''
    #
    lst_dir_raw_data = get_file_name(dir_dataset)
    for i, dir_raw_data in enumerate(lst_dir_raw_data):
        lst_new_save_data = []
        # open doc
        with open(dir_raw_data, 'rb') as file:
            lst_raw_data = pickle.load(file)
        #
        index_name = dir_raw_data.find('name_')
        index_pickle = dir_raw_data.find('.pickle')
        name_subfolder = dir_raw_data[len(root_dir_dataset)+1:index_name]
        if not os.path.exists(dir_save_new + name_subfolder):
            os.makedirs(dir_save_new + name_subfolder)
        #
        for cnt_frame in range(0, len(lst_raw_data) - cfg_info['dataset_separate']['frames_new_save'] + 1):
            dir_new_data = dir_save_new + name_subfolder + dir_raw_data[index_name:index_pickle] + '_frame_' + str(cnt_frame).zfill(3) + '.pickle'

            if cfg_info['dataset_separate']['frames_new_save'] > 1:
                lst_data_save = []
                for cnt_new_frame in range(0, cfg_info['dataset_separate']['frames_new_save']):
                    lst_data_save.append(lst_raw_data[cnt_frame+cnt_new_frame])
                with open(dir_new_data, 'wb') as f:
                    pickle.dump(lst_data_save, f)
                print("Separated frames are saved. TIME:", datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
            else:
                data_save = lst_raw_data[cnt_frame]
                ''''''
                with open(dir_new_data, 'wb') as f:
                    pickle.dump(data_save, f)
                print("Separated frame is saved. TIME:", datetime.datetime.now().strftime('%Y%m%d%H%M%S'))


def main():
    str_curr_time = datetime.datetime.now().strftime('%H%M%S%f')
    filedir_yaml = home_dir +"/TAMSER/TAMSER_python/00_dataset_collection/cfg/08_dataset_separate.yaml"
    with open(filedir_yaml, 'r') as f:
        cfg_info = yaml.load(f.read(), Loader=yaml.FullLoader)

    root_dir_dataset = home_dir + "/dataSSD/tamser_tactile_dataset_unknown_new"
    lst_folders = get_folder_name(root_dir_dataset)
    #
    try:
        if cfg_info['dataset_separate']['dir_save_new'] == "None":
            dir_save_new = lst_folders[0]
        else:
            dir_save_new = cfg_info['dataset_separate']['dir_save_new']
            # delete the previous data if necessary
            if os.path.exists(dir_save_new) and cfg_info['dataset_separate']['flg_delete_previous_data'] is True:
                delete_folder(dir_save_new)
            if not os.path.exists(dir_save_new):
                os.makedirs(dir_save_new)
        for i, dir_dataset in enumerate(lst_folders):
            worker(dir_dataset, dir_save_new, cfg_info, root_dir_dataset)
    #
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received. Stopping communication.")

    finally:
        print("Done.")


if __name__ == '__main__':
    main()





