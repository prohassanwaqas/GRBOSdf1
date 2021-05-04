import matplotlib

'''%matplotlib inline'''  # this is used in IPYTHON with jupyter notebook, thats why its not working with regualr python

import numpy as np

import glob
import pandas as pd

import Leap
import ctypes
import os
import sys
import pickle

# from leap_data_helper import *
import matplotlib.pyplot as plt

'''Leap.resample('M').sum().plot(kind="bar") #########
plt.show() ##################
'''



#####################################################

def load_data(data_files, n_disgard=50):
    data_file_names = glob.glob(data_files)
    data_file_names.sort()
    return data_file_names[n_disgard:]


#####################################################

person_id = 'p_0'
gesture_id = 'd'
num_disgard = 50

frame_leap_names = load_data('./' + person_id + '/' + gesture_id + '/leap_frames/*.data', num_disgard)


######################################################

def cal_2vec_angle(v1, v2):
    # return the value of cos(angle)
    return np.dot(v1, v2) / np.linalg.norm(v1) / np.linalg.norm(v2)


def get_frame(filename):
    frame = Leap.Frame()
    # filename = 'cali_1495553181022000.data'
    with open(os.path.realpath(filename), 'rb') as data_file:
        data = data_file.read()

    leap_byte_array = Leap.byte_array(len(data))
    address = leap_byte_array.cast().__long__()
    ctypes.memmove(address, data, len(data))

    frame.deserialize((leap_byte_array, len(data)))

    return frame


# Let's have a quick validation of the frame data
def get_joints(frame):
    joints = []
    for hand in frame.hands:
        for finger in hand.fingers:
            for b in range(0, 4):
                bone = finger.bone(b)
                joint_pos = bone.next_joint.to_float_array()
                joints.append(joint_pos)

    return np.array(joints)


def get_angles(frame):
    angles = []
    # palm position
    J0 = np.array(frame.hands[0].palm_position.to_float_array())
    # thumb
    J1 = np.array(frame.hands[0].fingers[0].bone(1).next_joint.to_float_array())
    J2 = np.array(frame.hands[0].fingers[0].bone(2).next_joint.to_float_array())
    J3 = np.array(frame.hands[0].fingers[0].bone(3).next_joint.to_float_array())
    # index
    J4 = np.array(frame.hands[0].fingers[1].bone(0).next_joint.to_float_array())
    J5 = np.array(frame.hands[0].fingers[1].bone(1).next_joint.to_float_array())
    J6 = np.array(frame.hands[0].fingers[1].bone(2).next_joint.to_float_array())
    J7 = np.array(frame.hands[0].fingers[1].bone(3).next_joint.to_float_array())
    # middle
    J8 = np.array(frame.hands[0].fingers[2].bone(0).next_joint.to_float_array())
    J9 = np.array(frame.hands[0].fingers[2].bone(1).next_joint.to_float_array())
    J10 = np.array(frame.hands[0].fingers[2].bone(2).next_joint.to_float_array())
    J11 = np.array(frame.hands[0].fingers[2].bone(3).next_joint.to_float_array())
    # ring
    J12 = np.array(frame.hands[0].fingers[3].bone(0).next_joint.to_float_array())
    J13 = np.array(frame.hands[0].fingers[3].bone(1).next_joint.to_float_array())
    J14 = np.array(frame.hands[0].fingers[3].bone(2).next_joint.to_float_array())
    J15 = np.array(frame.hands[0].fingers[3].bone(3).next_joint.to_float_array())
    # pinky
    J16 = np.array(frame.hands[0].fingers[4].bone(0).next_joint.to_float_array())
    J17 = np.array(frame.hands[0].fingers[4].bone(1).next_joint.to_float_array())
    J18 = np.array(frame.hands[0].fingers[4].bone(2).next_joint.to_float_array())
    J19 = np.array(frame.hands[0].fingers[4].bone(3).next_joint.to_float_array())

    # A1-4
    A = cal_2vec_angle((J1 - J0), (J4 - J0))
    angles.append(A)
    A = cal_2vec_angle((J4 - J0), (J8 - J0))
    angles.append(A)
    A = cal_2vec_angle((J8 - J0), (J12 - J0))
    angles.append(A)
    A = cal_2vec_angle((J12 - J0), (J16 - J0))
    angles.append(A)

    # A5,6 on thumb
    A = cal_2vec_angle((J2 - J1), (J1 - J0))
    angles.append(A)
    A = cal_2vec_angle((J3 - J2), (J2 - J1))
    angles.append(A)

    # A7-9 on index
    A = cal_2vec_angle((J5 - J4), (J4 - J0))
    angles.append(A)
    A = cal_2vec_angle((J6 - J5), (J5 - J4))
    angles.append(A)
    A = cal_2vec_angle((J7 - J6), (J6 - J5))
    angles.append(A)

    # A10-12 on middle
    A = cal_2vec_angle((J9 - J8), (J8 - J0))
    angles.append(A)
    A = cal_2vec_angle((J10 - J9), (J9 - J8))
    angles.append(A)
    A = cal_2vec_angle((J11 - J10), (J10 - J9))
    angles.append(A)

    # A13-15 on ring
    A = cal_2vec_angle((J13 - J12), (J12 - J0))
    angles.append(A)
    A = cal_2vec_angle((J14 - J13), (J13 - J12))
    angles.append(A)
    A = cal_2vec_angle((J15 - J14), (J14 - J13))
    angles.append(A)

    # A16-18 on pinky
    A = cal_2vec_angle((J17 - J16), (J16 - J0))
    angles.append(A)
    A = cal_2vec_angle((J18 - J17), (J17 - J16))
    angles.append(A)
    A = cal_2vec_angle((J19 - J18), (J18 - J17))
    angles.append(A)

    # A19-22 between adjacent finger tips
    A = cal_2vec_angle((J3 - J2), (J7 - J6))
    angles.append(A)
    A = cal_2vec_angle((J7 - J6), (J11 - J10))
    angles.append(A)
    A = cal_2vec_angle((J11 - J10), (J15 - J14))
    angles.append(A)
    A = cal_2vec_angle((J15 - J14), (J19 - J18))
    angles.append(A)

    return np.array(angles)


######################################################

person_id_list = ['p_0', 'p_1', 'p_2', 'p_3', 'p_4']
gesture_id_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k', 'l', 'm',
                   'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y']

## Initialization

features_angles_list = []
label_list = []

# (This is a must) Create a controller to connect to the Leap Motion device
controller = Leap.Controller()

for person_id_i, person_id in enumerate(person_id_list):
    for gesture_i, gesture_id in enumerate(gesture_id_list):

        frame_leap_names = load_data('./' + person_id + '/' + gesture_id + '/leap_frames/*.data')

        # extract the angle features
        for frame_name in frame_leap_names:
            frame = get_frame(frame_name)
            features_angles_list.append(get_angles(frame))

#print ((person_id + ': ' + gesture_id + ' has {} samples, label: {}').format(len(img_leap_l_names), [person_id_i, gesture_i]) )


#print ((person_id + ':' + gesture_id + 'has {} samples, label: {}')[person_id_i, gesture_i])


#print(person_id + ':' + gesture_id + 'has {} samples, label: {}').format([person_id_i, gesture_i])

print((person_id + ": " + gesture_id + "has {} samples, label: {}").format(person_id, gesture_id))
######################################################

# Save a dictionary into a pickle file.
pickle.dump({'features_angles': np.array(features_angles_list),
            # 'labels': np.array(label_list)}, open("./datasets/dataset.p", "wb"), )
            'labels': np.array(label_list)}, open("C:\Users\hassan\PycharmProjects\GRBOSdf1\dataset\dataset.p", "wb"),)

######################################################

#object = pd.read_pickle(r'C:\Users\hassan\PycharmProjects\GRBOSdf1\dataset\dataset.p')