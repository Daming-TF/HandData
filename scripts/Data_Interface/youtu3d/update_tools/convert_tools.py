import os
import numpy as np


def get_file_list(file_path):
    dir_list = os.listdir(file_path)
    if not dir_list:
        return
    else:
        dir_list = sorted(dir_list, key=lambda x: os.path.getmtime(os.path.join(file_path, x)))
        return dir_list


def get_keypoints(label_feature):
    # init
    handlandmarks_list = []
    hand1 = np.zeros((21, 3))
    hand2 = np.zeros((21, 3))

    # Convert format
    keys = label_feature[0].keys()
    for key in keys:
        hand_info = key.split('-')
        hand_index = int(hand_info[0])
        landmark_index = int(hand_info[1])
        if hand_index == 0:
            hand1[landmark_index, :2] = np.array(label_feature[0][key][:2])
            hand1[landmark_index, 2] = 2
        elif hand_index == 1:
            hand2[landmark_index, :2] = np.array(label_feature[0][key][:2])
            hand2[landmark_index, 2] = 2
    if (not np.all(hand1 == 0)) and check_keypoints(hand1):
        handlandmarks_list.append(hand1)
    if (not np.all(hand2 == 0)) and check_keypoints(hand2):
        handlandmarks_list.append(hand2)

    # return list
    return handlandmarks_list


def check_keypoints(keypoints):
    kps_valid_bool = keypoints[:, -1].astype(np.bool)
    key_pts = keypoints[:, :2][kps_valid_bool]
    if np.shape(key_pts)[0] > 16:
        return 1
    else:
        return 0