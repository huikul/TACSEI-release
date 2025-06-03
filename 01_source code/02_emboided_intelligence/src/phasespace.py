"""
created by
"""
import sys, os
home_dir = os.environ['HOME']
sys.path.append(home_dir + '/ws_tracking/devel/lib/python3/dist-packages')  # adjust the package rout if necessary
import numpy as np

import rospy
# import ros_numpy
from phasespace_msgs.msg import MarkersStamped


class ROS_PhaseSpace_Listener(object):
    def __init__(self, name_node='ros_node_zjd_pc_phasespece', flg_ROS_init=False):
        if flg_ROS_init:
            self.node_listener = rospy.init_node(name_node, anonymous=True)
            self.rate = rospy.Rate(60)  # 30hz

    def get_pose_makers(self):
        makers_data = rospy.wait_for_message('/ps_owl/markers', MarkersStamped)
        # print(type(makers_data))
        lst_pose = []
        lst_name = []
        lst_id = []
        for cnt_pt, msg in enumerate(makers_data.data):
            lst_pose.append(np.array([makers_data.data[cnt_pt].position.x,
                                      makers_data.data[cnt_pt].position.y,
                                      makers_data.data[cnt_pt].position.z], dtype=np.float32))
            lst_name.append(makers_data.data[cnt_pt].name)
            lst_id.append(makers_data.data[cnt_pt].id)
            # pass
        return lst_pose, lst_name, lst_id


if __name__ == '__main__':
    rospy.init_node('test_zjd_pc', anonymous=True)
    test_node = ROS_PhaseSpace_Listener()
    lst_pose, lst_name, lst_id = test_node.get_pose_makers()
    print("lst_pose: ", lst_pose)
    print("lst_name: ", lst_name)
    print("lst_id: ", lst_id)




