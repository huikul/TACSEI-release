"""
created by XXXX
18.10.2024  21:02

email:

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


def smooth_air_pressure(start_press=[0], end_press=[100], step=10):
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
    # test
    start = [0, 10, 20]
    end = [100, 100, 100]
    smooth_air_pressure(end, start, step=30)
    #
    time.sleep(1)
    str_curr_time = datetime.datetime.now().strftime('%H%M%S%f')
    fildir_yaml = home_dir + "/dataSSD/TAMSER/TAMSER_python/00_dataset_collection/cfg/04_dataset_collection_3f_static.yaml"
    with open(fildir_yaml, 'r') as f:
        cfg_info = yaml.load(f.read(), Loader=yaml.FullLoader)

    if not os.path.exists(cfg_info['dataset_collection']['save_dir']):
        os.makedirs(cfg_info['dataset_collection']['save_dir'])
    #
    str_cmd_fing_open = "fing_ctrl(0, 0, 0)\r\n"
    str_cmd_read_tactile = "test_command(6)\r\n"
    # str_cmd_read_tactile = "read_tac_fun(0)\r\n"
    press_fing_1 = int(cfg_info['dataset_collection']['air_fing'])
    press_fing_2 = press_fing_1
    press_fing_3 = press_fing_1
    # str_cmd_fing_ctrl = "fing_ctrl({}, {}, {})\r\n".format(press_fing_1, press_fing_2, press_fing_3)
    lst_cmd_fing_ctrl = smooth_air_pressure(start_press=[0, 0, 0],
                                            end_press=[press_fing_1, press_fing_2, press_fing_3],
                                            step=30)
    #
    serial_port = SerialPort(cfg_info['serial_port']['port'], cfg_info['serial_port']['baudrate'])
    serial_port.open()

    rospy.init_node('ros_zjd_pc', anonymous=True)
    tracking_node = ROS_PhaseSpace_Listener()

    time.sleep(0.5)
    lst_pose_init, lst_marker_name, _ = tracking_node.get_pose_makers()
    serial_port.send_message(str_cmd_read_tactile)
    tac_data_init = serial_port.receive_tactile_data()
    if len(lst_pose_init) < cfg_info['dataset_collection']['num_markers'] or \
            len(lst_marker_name) < cfg_info['dataset_collection']['num_markers']:
        return

    #
    try:
        # Sending a message
        # str_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
        lst_info_save = []
        #
        # serial_port.send_message(str_cmd_fing_ctrl)
        for i, cmd in enumerate(lst_cmd_fing_ctrl):
            serial_port.send_message(cmd)
            response = serial_port.receive_message()
            if response:
                print(f"Received: {response}")
            time.sleep(0.02)
        # Receiving a message
        ''''''
        # response = serial_port.receive_message()  # receive
        response = serial_port.receive_message()
        if response:
            print(f"Received: {response}")
        #
        #
        start_time = time.time()
        for i in range(0, cfg_info['dataset_collection']['max_loop']):
            info_frame = {}
            serial_port.send_message(str_cmd_read_tactile)
            tac_data = serial_port.receive_tactile_data()
            lst_pose, lst_marker_name, _ = tracking_node.get_pose_makers()
            curr_time = time.time()
            # tac_data = serial_port.receive_tactile_long()
            lst_info_save.append({'name_obj': cfg_info['dataset_collection']['name_obj'],
                                  'size_obj': cfg_info['dataset_collection']['size_obj'],
                                  'maker_name': lst_marker_name,
                                  'marker_poses_init': {'raw': lst_pose_init},
                                  'tactile_data_init': {'raw': tac_data_init},
                                  'marker_poses_curr': {'raw': lst_pose},
                                  'tactile_data_curr': {'raw': tac_data},
                                  'fing_press': {'raw': np.array([press_fing_1, press_fing_2, press_fing_3], dtype=np.float32)},
                                  'time_stamp': curr_time - start_time})

            # time.sleep(time_gap)
        #
        str_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        file_name = "/name_{}_sz_{}_pr1_{}_pr2_{}_pr3_{}_time_{}".format(cfg_info['dataset_collection']['name_obj'],
                                                                         cfg_info['dataset_collection']['size_obj'],
                                                                         press_fing_1, press_fing_2, press_fing_3, str(str_time))
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
            # c = lst_info_save[i]['tactile_data']
            # a = lst_info_save[i]['tactile_data'].max()
            # b = cfg_info['dataset_collection']['max_tac_value']
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
        # str_cmd_fing_open = "fing_ctrl(0, 0, 0)\r\n"
        # serial_port.send_message(str_cmd_fing_open)
        lst_cmd_fing_ctrl = smooth_air_pressure(start_press=[press_fing_1, press_fing_2, press_fing_3],
                                                end_press=[0, 0, 0],
                                                step=30)
        for i, cmd in enumerate(lst_cmd_fing_ctrl):
            serial_port.send_message(cmd)
            response = serial_port.receive_message()
            time.sleep(0.02)
        serial_port.close()

    print("Done.")


    """
    lst_pose, lst_name, lst_id = test_node.get_pose_makers()
    print("lst_pose: ", lst_pose)
    print("lst_name: ", lst_name)
    print("lst_id: ", lst_id)
    """


if __name__ == '__main__':
    main()





