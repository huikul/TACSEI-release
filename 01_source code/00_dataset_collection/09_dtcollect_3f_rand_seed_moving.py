"""
created by

"""
import os, time, sys
home_dir = os.environ['HOME']
sys.path.append(home_dir + '/motion_tracking_ws/devel/lib/python3/dist-packages')
sys.path.append(home_dir + '/ws_tracking/devel/lib/python3/dist-packages')
sys.path.append(home_dir + 'dataSSD/ws_ros/devel/lib/python3/dist-packages')
sys.path.append('/opt/ros/noetic/lib/python3/dist-packages')

import rospy
import datetime
import numpy as np
import yaml

from src.phasespace import ROS_PhaseSpace_Listener
from src.serial_comm import SerialPort
import pickle
import random


def smooth_air_pressure(start_press=None, end_press=None, step=10):
    arrs_pressure = []
    max_length = 0
    # Generate arrays and find the maximum length
    for start, end in zip(start_press, end_press):
        arr_pressure = np.linspace(int(start), int(end), step, dtype=int)
        if end not in arr_pressure:
            arr_pressure = np.append(arr_pressure, int(end))
        arrs_pressure.append(arr_pressure)

    lst_cmd_fing_ctrl = []
    for i in range(0, step):
        lst_cmd_fing_ctrl.append("fing_ctrl({}, {}, {})\r\n".format(arrs_pressure[0][i],
                                                                    arrs_pressure[1][i],
                                                                    arrs_pressure[2][i]))
    return lst_cmd_fing_ctrl


def main():
    str_curr_time = datetime.datetime.now().strftime('%H%M%S%f')
    fildir_yaml = home_dir +"/dataSSD/TAMSER/TAMSER_python/00_dataset_collection/cfg/09_dataset_collection_3f_rand_seed_moving.yaml"
    with open(fildir_yaml, 'r') as f:
        cfg_info = yaml.load(f.read(), Loader=yaml.FullLoader)

    if not os.path.exists(cfg_info['dataset_collection']['save_dir']):
        os.makedirs(cfg_info['dataset_collection']['save_dir'])
    #
    if 'seed_rand' in cfg_info['dataset_collection']:
        if cfg_info['dataset_collection']['seed_rand'] > 0:
            random.seed(int(cfg_info['dataset_collection']['seed_rand']))
    #
    str_cmd_fing_open = "fing_ctrl(0, 0, 0)\r\n"
    str_cmd_read_tactile = "test_command(6)\r\n"
    # str_cmd_read_tactile = "read_tac_fun(0)\r\n"
    air_th_pre = 0
    air_ff_pre = 0
    air_mf_pre = 0
    str_cmd_fing_ctrl = "fing_ctrl({}, {}, {})\r\n".format(air_th_pre, air_ff_pre, air_mf_pre)
    #
    serial_port = SerialPort(cfg_info['serial_port']['port'], cfg_info['serial_port']['baudrate'])
    serial_port.open()
    time.sleep(8.0)
    serial_port.send_message(str_cmd_fing_ctrl)
    serial_port.receive_message()  # Receiving a message

    rospy.init_node('ros_zjd_pc', anonymous=True)
    tracking_node = ROS_PhaseSpace_Listener()

    time.sleep(0.5)
    lst_pose_init, lst_marker_name, _ = tracking_node.get_pose_makers()
    serial_port.send_message(str_cmd_read_tactile)
    tac_data_init = serial_port.receive_tactile_data()
    if len(lst_pose_init) < cfg_info['dataset_collection']['num_markers'] or \
            len(lst_marker_name) < cfg_info['dataset_collection']['num_markers']:

        print("Number of markers: ", len(lst_pose_init))
        print("Names of markers: ", lst_marker_name)
        return

    #
    air_th_pre = 0
    air_ff_pre = 0
    air_mf_pre = 0
    air_th_curr = 0
    air_ff_curr = 0
    air_mf_curr = 0

    air_th_curr = int(cfg_info['dataset_collection']['min_air'])
    air_ff_curr = int(air_th_curr)
    air_mf_curr = int(air_th_curr)

    lst_cmd_fing_ctrl = smooth_air_pressure(start_press=[air_th_pre, air_ff_pre, air_mf_pre],
                                            end_press=[air_th_curr, air_ff_curr, air_mf_curr],
                                            step=30)
    air_th_pre = int(air_th_curr)
    air_ff_pre = int(air_ff_curr)
    air_mf_pre = int(air_mf_curr)

    for i, cmd in enumerate(lst_cmd_fing_ctrl):
        serial_port.send_message(cmd)
        response = serial_port.receive_message()
        time.sleep(0.02)

    try:
        max_iter_air_change = int(cfg_info['dataset_collection']['max_iter_air_change'])
        for cnt_air_change in range(0, max_iter_air_change):
            # Sending a message
            # str_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
            lst_info_save = []
            air_th_curr = random.randint(cfg_info['dataset_collection']['min_air'], cfg_info['dataset_collection']['max_air'])
            air_ff_curr = random.randint(cfg_info['dataset_collection']['min_air'], cfg_info['dataset_collection']['max_air'])
            air_mf_curr = random.randint(cfg_info['dataset_collection']['min_air'], cfg_info['dataset_collection']['max_air'])

            lst_cmd_fing_ctrl = smooth_air_pressure(start_press=[air_th_pre, air_ff_pre, air_mf_pre],
                                                    end_press=[air_th_curr, air_ff_curr, air_mf_curr],
                                                    step=20)
            air_th_pre = int(air_th_curr)
            air_ff_pre = int(air_ff_curr)
            air_mf_pre = int(air_mf_curr)

            for i, cmd in enumerate(lst_cmd_fing_ctrl):
                serial_port.send_message(cmd)
                response = serial_port.receive_message()
                time.sleep(0.02)
            time.sleep(2.0)
            # Receiving a message
            #
            info_frame = {}
            #
            start_time = time.time()
            for i in range(0, cfg_info['dataset_collection']['loop_per_airsetp']):
                serial_port.send_message(str_cmd_read_tactile)
                tac_data = serial_port.receive_tactile_data()
                lst_pose, lst_marker_name, _ = tracking_node.get_pose_makers()
                print("tac_data", tac_data[0,:,:], tac_data[1,:,:], tac_data[2,:,:])
                print(len(lst_marker_name), lst_marker_name)
                print("cnt_air_change: ", cnt_air_change)
                curr_time = time.time()
                # tac_data = serial_port.receive_tactile_long()
                lst_info_save.append({'name_obj': cfg_info['dataset_collection']['name_obj'],
                                      'size_obj': cfg_info['dataset_collection']['size_obj'],
                                      'maker_name': lst_marker_name,
                                      'marker_poses_init': {'raw': lst_pose_init},
                                      'tactile_data_init': {'raw': tac_data_init},
                                      'marker_poses_curr': {'raw': lst_pose},
                                      'tactile_data_curr': {'raw': tac_data},
                                      'fing_press': {'raw': np.array([air_th_curr, air_ff_curr, air_mf_curr], dtype=np.float32)},
                                      'time_stamp': curr_time - start_time})
                # time.sleep(time_gap)
            #
            str_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
            file_name = "/name_{}_sz_{}_pr1_{}_pr2_{}_pr3_{}_time_{}".format(cfg_info['dataset_collection']['name_obj'],
                                                                             cfg_info['dataset_collection']['size_obj'],
                                                                             air_th_curr, air_ff_curr, air_mf_curr, str(str_time))
            # check completeness
            for i in range(len(lst_info_save) - 1, -1, -1):
                # delete incomplete data
                if len(lst_info_save[i]['maker_name']) < cfg_info['dataset_collection']['num_markers'] or \
                        len(lst_info_save[i]['marker_poses_init']['raw']) < cfg_info['dataset_collection']['num_markers'] or \
                        len(lst_info_save[i]['marker_poses_curr']['raw']) < cfg_info['dataset_collection']['num_markers'] or \
                        lst_info_save[i]['tactile_data_curr']['raw'].shape[0] < cfg_info['dataset_collection']['num_fing_press'] or \
                        lst_info_save[i]['tactile_data_curr']['raw'].shape[1] < cfg_info['dataset_collection']['row_tac_array'] or \
                        lst_info_save[i]['tactile_data_curr']['raw'].shape[2] < cfg_info['dataset_collection']['col_tac_array'] or \
                        lst_info_save[i]['fing_press']['raw'].size < cfg_info['dataset_collection']['num_fing_press']:
                    del lst_info_save[i]
                    print("delete incomplete data")
                    continue
                # delete extreme data
                if lst_info_save[i]['tactile_data_curr']['raw'].max() > cfg_info['dataset_collection']['max_tac_value'] or \
                        lst_info_save[i]['tactile_data_curr']['raw'].min() < cfg_info['dataset_collection']['min_tac_value']:
                    del lst_info_save[i]
                    print("delete extreme data")
                    continue
            # do not save the data if not long enough
            if len(lst_info_save) < cfg_info['dataset_collection']['min_len_data']:
                print("The data are too short")
            else:
                with open(cfg_info['dataset_collection']['save_dir'] + file_name + '.pickle', 'wb') as f:
                    pickle.dump(lst_info_save, f)

    except KeyboardInterrupt:
        print("\nKeyboard interrupt received. Stopping communication.")

    finally:
        # Ensure the serial port is closed before exiting
        lst_cmd_fing_ctrl = smooth_air_pressure(start_press=[air_th_curr, air_ff_curr, air_mf_curr],
                                                end_press=[0, 0, 0],
                                                step=50)
        for i, cmd in enumerate(lst_cmd_fing_ctrl):
            serial_port.send_message(cmd)
            response = serial_port.receive_message()
            time.sleep(0.02)
        serial_port.close()

    print("Done.")


if __name__ == '__main__':
    main()





