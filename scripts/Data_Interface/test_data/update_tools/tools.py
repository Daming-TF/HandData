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
    handlandmarks_list = []
    hand1 = np.zeros((21, 3))
    hand2 = np.zeros((21, 3))
    keys = label_feature[0].keys()
    for key in keys:
        hand_info = key.split('-')
        hand_index = int(hand_info[0])
        landmark_index = int(hand_info[1])
        if hand_index == 0:
            hand1[landmark_index, :2] = np.array(label_feature[0][key][:2])
        elif hand_index == 1:
            hand2[landmark_index, :2] = np.array(label_feature[0][key][:2])
    if not np.all(hand1 == 0):
        hand1[:, 2] = 2
        coco_kps = hand1.flatten().tolist()
        handlandmarks_list.append(coco_kps)
    if not np.all(hand2 == 0):
        hand2[:, 2] = 2
        coco_kps = hand2.flatten().tolist()
        handlandmarks_list.append(coco_kps)
    return handlandmarks_list
