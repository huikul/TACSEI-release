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
from src.docs_proc import get_file_name, get_folder_name


def worker(dir_dataset, cfg_info):
    if cfg_info['dataset_normalization']['dir_save_new'] == "None":
        dir_save_new = dir_dataset
    else:
        dir_save_new = cfg_info['dataset_normalization']['dir_save_new']
        if not os.path.exists(dir_save_new):
            os.makedirs(dir_save_new)
    #
    lst_dir_raw_data = get_file_name(dir_dataset)
    for i, dir_raw_data in enumerate(lst_dir_raw_data):
        lst_new_save_data = []
        flg_update = False
        # open doc
        with open(dir_raw_data, 'rb') as file:
            lst_raw_data = pickle.load(file)
        # norm
        for cnt_frame in range(0, len(lst_raw_data)):
            raw_data = lst_raw_data[cnt_frame]
            if 'norm' in raw_data['marker_poses_init'] and cfg_info['dataset_normalization']['flg_force_refresh'] is False:
                print("Normalization was done, skip the current data. TIME: ", datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
                flg_update = False
                break
            else:
                flg_update = True
                # sort the marker name based on the ascii code. After sorting, ff1, ff2,...,ff5, mf1, ..., mf5, palm, th1, th2, ...
                sorted_with_indices = sorted(enumerate(raw_data['maker_name']), key=lambda x: x[1])
                sorted_indices = [index for index, value in sorted_with_indices]
                sorted_strings = [value for index, value in sorted_with_indices]
                # sort the corresponding marker poses
                sorted_marker_poses = np.array([raw_data['marker_poses_init']['raw'][i] for i in sorted_indices])
                # norm all poses
                # poses 不需要 norm
                '''
                sorted_marker_poses = (sorted_marker_poses - cfg_info['dataset_normalization']['min_marker_pos']) / (
                            cfg_info['dataset_normalization']['max_marker_pos'] - cfg_info['dataset_normalization']['min_marker_pos'])
                '''
                # separate
                pt_palm_raw = np.array(sorted_marker_poses[10], dtype=np.float32)
                pts_th_raw = np.array(sorted_marker_poses[11:16], dtype=np.float32)
                pts_ff_raw = np.array(sorted_marker_poses[0:5], dtype=np.float32)
                pts_mf_raw = np.array(sorted_marker_poses[5:10], dtype=np.float32)
                raw_data['marker_poses_init']['raw'] = {'pt_palm': None, 'pts_ff': None, 'pts_mf': None,
                                                         # 'max': cfg_info['dataset_normalization']['max_marker_pos'],
                                                         # 'min': cfg_info['dataset_normalization']['min_marker_pos'],
                                                         'note': 'th1, th2, ...'}
                raw_data['marker_poses_init']['raw']['pt_palm'] = pt_palm_raw
                raw_data['marker_poses_init']['raw']['pts_th'] = pts_th_raw
                raw_data['marker_poses_init']['raw']['pts_ff'] = pts_ff_raw
                raw_data['marker_poses_init']['raw']['pts_mf'] = pts_mf_raw

                # similar method for current pose
                sorted_marker_poses = np.array([raw_data['marker_poses_curr']['raw'][i] for i in sorted_indices])
                pt_palm_raw = np.array(sorted_marker_poses[10], dtype=np.float32)
                pts_th_raw = np.array(sorted_marker_poses[11:16], dtype=np.float32)
                pts_ff_raw = np.array(sorted_marker_poses[0:5], dtype=np.float32)
                pts_mf_raw = np.array(sorted_marker_poses[5:10], dtype=np.float32)
                raw_data['marker_poses_curr']['raw'] = {'pt_palm': None, 'pts_ff': None, 'pts_mf': None,
                                                         'note': 'th1, th2, ...'}
                raw_data['marker_poses_curr']['raw']['pt_palm'] = pt_palm_raw
                raw_data['marker_poses_curr']['raw']['pts_th'] = pts_th_raw
                raw_data['marker_poses_curr']['raw']['pts_ff'] = pts_ff_raw
                raw_data['marker_poses_curr']['raw']['pts_mf'] = pts_mf_raw
                #
                # save the offset: pts_fingers - pt_palm
                raw_data['marker_poses_offset2palm'] = {'raw': {'pt_palm': None, 'pts_th': None, 'pts_ff': None, 'pts_mf': None},
                                                        'norm': {'pt_palm': None, 'pts_th': None, 'pts_ff': None, 'pts_mf': None,
                                                                 'max': cfg_info['dataset_normalization']['max_marker_pos'],
                                                                 'min': cfg_info['dataset_normalization']['min_marker_pos']},
                                                        'note': 'pts_fingers - pt_palm'}
                raw_data['marker_poses_offset2palm']['raw']['pt_palm'] = raw_data['marker_poses_curr']['raw']['pt_palm'] - raw_data['marker_poses_curr']['raw']['pt_palm']
                raw_data['marker_poses_offset2palm']['raw']['pts_th'] = raw_data['marker_poses_curr']['raw']['pts_th'] - raw_data['marker_poses_curr']['raw']['pt_palm']
                raw_data['marker_poses_offset2palm']['raw']['pts_ff'] = raw_data['marker_poses_curr']['raw']['pts_ff'] - raw_data['marker_poses_curr']['raw']['pt_palm']
                raw_data['marker_poses_offset2palm']['raw']['pts_mf'] = raw_data['marker_poses_curr']['raw']['pts_mf'] - raw_data['marker_poses_curr']['raw']['pt_palm']
                raw_data['marker_poses_offset2palm']['norm']['pt_palm'] = (raw_data['marker_poses_offset2palm']['raw']['pt_palm'] - raw_data['marker_poses_offset2palm']['norm']['min']) / (raw_data['marker_poses_offset2palm']['norm']['max'] - raw_data['marker_poses_offset2palm']['norm']['min'])
                raw_data['marker_poses_offset2palm']['norm']['pts_th'] = (raw_data['marker_poses_offset2palm']['raw']['pts_th'] - raw_data['marker_poses_offset2palm']['norm']['min']) / (raw_data['marker_poses_offset2palm']['norm']['max'] - raw_data['marker_poses_offset2palm']['norm']['min'])
                raw_data['marker_poses_offset2palm']['norm']['pts_ff'] = (raw_data['marker_poses_offset2palm']['raw']['pts_ff'] - raw_data['marker_poses_offset2palm']['norm']['min']) / (raw_data['marker_poses_offset2palm']['norm']['max'] - raw_data['marker_poses_offset2palm']['norm']['min'])
                raw_data['marker_poses_offset2palm']['norm']['pts_mf'] = (raw_data['marker_poses_offset2palm']['raw']['pts_mf'] - raw_data['marker_poses_offset2palm']['norm']['min']) / (raw_data['marker_poses_offset2palm']['norm']['max'] - raw_data['marker_poses_offset2palm']['norm']['min'])
                # save the offset: pts_curr - pts_init
                raw_data['marker_poses_offset2self'] = {'raw': {'pt_palm': None, 'pts_th': None, 'pts_ff': None, 'pts_mf': None},
                                                        'norm': {'pt_palm': None, 'pts_th': None, 'pts_ff': None, 'pts_mf': None,
                                                                 'max': cfg_info['dataset_normalization']['max_marker_pos'],
                                                                 'min': cfg_info['dataset_normalization']['min_marker_pos']},
                                                        'note': 'pts_curr - pts_init'}
                raw_data['marker_poses_offset2self']['raw']['pt_palm'] = raw_data['marker_poses_curr']['raw']['pt_palm'] - raw_data['marker_poses_init']['raw']['pt_palm']
                raw_data['marker_poses_offset2self']['raw']['pts_th'] = raw_data['marker_poses_curr']['raw']['pts_th'] - raw_data['marker_poses_init']['raw']['pts_th']
                raw_data['marker_poses_offset2self']['raw']['pts_ff'] = raw_data['marker_poses_curr']['raw']['pts_ff'] - raw_data['marker_poses_init']['raw']['pts_ff']
                raw_data['marker_poses_offset2self']['raw']['pts_mf'] = raw_data['marker_poses_curr']['raw']['pts_mf'] - raw_data['marker_poses_init']['raw']['pts_mf']
                raw_data['marker_poses_offset2self']['norm']['pt_palm'] = (raw_data['marker_poses_offset2self']['raw']['pt_palm'] - raw_data['marker_poses_offset2self']['norm']['min']) / (raw_data['marker_poses_offset2self']['norm']['max'] - raw_data['marker_poses_offset2self']['norm']['min'])
                raw_data['marker_poses_offset2self']['norm']['pts_th'] = (raw_data['marker_poses_offset2self']['raw']['pts_th'] - raw_data['marker_poses_offset2self']['norm']['min']) / (raw_data['marker_poses_offset2self']['norm']['max'] - raw_data['marker_poses_offset2self']['norm']['min'])
                raw_data['marker_poses_offset2self']['norm']['pts_ff'] = (raw_data['marker_poses_offset2self']['raw']['pts_ff'] - raw_data['marker_poses_offset2self']['norm']['min']) / (raw_data['marker_poses_offset2self']['norm']['max'] - raw_data['marker_poses_offset2self']['norm']['min'])
                raw_data['marker_poses_offset2self']['norm']['pts_mf'] = (raw_data['marker_poses_offset2self']['raw']['pts_mf'] - raw_data['marker_poses_offset2self']['norm']['min']) / ( raw_data['marker_poses_offset2self']['norm']['max'] - raw_data['marker_poses_offset2self']['norm']['min'])

                # norm tactile
                raw_data['tactile_data_init']['norm'] = {'th': None, 'ff': None, 'mf': None,
                                                         'max': cfg_info['dataset_normalization']['max_tac_value'],
                                                         'min': cfg_info['dataset_normalization']['min_tac_value'],
                                                         'note': 'from fingertip to root, them chamber, revert from max to min'}
                raw_data['tactile_data_init']['norm']['th'] = 1.0 - (raw_data['tactile_data_init']['raw'][0, :, :] - raw_data['tactile_data_init']['norm']['min']) / (raw_data['tactile_data_init']['norm']['max'] - raw_data['tactile_data_init']['norm']['min'])
                raw_data['tactile_data_init']['norm']['ff'] = 1.0 - (raw_data['tactile_data_init']['raw'][1, :, :] - raw_data['tactile_data_init']['norm']['min']) / (raw_data['tactile_data_init']['norm']['max'] - raw_data['tactile_data_init']['norm']['min'])
                raw_data['tactile_data_init']['norm']['mf'] = 1.0 - (raw_data['tactile_data_init']['raw'][2, :, :] - raw_data['tactile_data_init']['norm']['min']) / (raw_data['tactile_data_init']['norm']['max'] - raw_data['tactile_data_init']['norm']['min'])

                raw_data['tactile_data_curr']['norm'] = {'th': None, 'ff': None, 'mf': None,
                                                         'max': cfg_info['dataset_normalization']['max_tac_value'],
                                                         'min': cfg_info['dataset_normalization']['min_tac_value'],
                                                         'note': 'from fingertip to root, them chamber, revert from max to min'}
                raw_data['tactile_data_curr']['norm']['th'] = 1.0 - (raw_data['tactile_data_curr']['raw'][0, :, :] - raw_data['tactile_data_curr']['norm']['min']) / (raw_data['tactile_data_curr']['norm']['max'] - raw_data['tactile_data_curr']['norm']['min'])
                raw_data['tactile_data_curr']['norm']['ff'] = 1.0 - (raw_data['tactile_data_curr']['raw'][1, :, :] - raw_data['tactile_data_curr']['norm']['min']) / (raw_data['tactile_data_curr']['norm']['max'] - raw_data['tactile_data_curr']['norm']['min'])
                raw_data['tactile_data_curr']['norm']['mf'] = 1.0 - (raw_data['tactile_data_curr']['raw'][2, :, :] - raw_data['tactile_data_curr']['norm']['min']) / (raw_data['tactile_data_curr']['norm']['max'] - raw_data['tactile_data_curr']['norm']['min'])

                # norm air pressure
                raw_data['fing_press']['norm'] = {'values': None,
                                                  'max': cfg_info['dataset_normalization']['max_air_pressure'],
                                                  'min': cfg_info['dataset_normalization']['min_air_pressure'],
                                                  'note': 'th, ff, mf, ...'}
                raw_data['fing_press']['norm']['value'] = np.array((raw_data['fing_press']['raw'] - raw_data['fing_press']['norm']['min'])/(raw_data['fing_press']['norm']['max'] - raw_data['fing_press']['norm']['min']), dtype=np.float32)

                lst_new_save_data.append(raw_data)

        if flg_update is True:
            with open(dir_raw_data, 'wb') as f:
                pickle.dump(lst_new_save_data, f)
            print("Normalized data are saved. TIME:", datetime.datetime.now().strftime('%Y%m%d%H%M%S'))

def main():
    str_curr_time = datetime.datetime.now().strftime('%H%M%S%f')
    filedir_yaml = home_dir +"/TAMSER/TAMSER_python/00_dataset_collection/cfg/07_dataset_normalization.yaml"
    with open(filedir_yaml, 'r') as f:
        cfg_info = yaml.load(f.read(), Loader=yaml.FullLoader)

    root_dir_dataset = home_dir + "/dataSSD/tamser_tactile_dataset_unknown_new"
    lst_folders = get_folder_name(root_dir_dataset)
    #
    try:
        for i, dir_dataset in enumerate(lst_folders):
            worker(dir_dataset, cfg_info)
    #
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received. Stopping communication.")

    finally:
        print("Done.")


if __name__ == '__main__':
    main()





