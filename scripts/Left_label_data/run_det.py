'''
    通过旋转操作调正然后输入到封装的Mediapipe接口，得到左右手label
'''
import os
import cv2
from json_tools import make_json_head
import numpy as np
import json
from tqdm import tqdm
from rotation_tools import HandInfo, HandLandModel
from convert_coco_format import convert_coco_format_left_label
from copy import deepcopy
from tools import draw_2d_points

mode = 'test'
# E:\whole_body_data\annotations\person_keypoints_{mode}2017.json
json_dir = fr'E:\whole_body_data\annotations\person_keypoints_{mode}2017.json'
txt_dir = fr"E:\left_hand_label_data\detect\record_{mode}.txt"
SAVE_PATH = r"E:\left_hand_label_data\detect"
JSON_NAME = f'{mode}.json'
BADCASE_JSON_NAME = f'badcase_{mode}.json'
num_joints = 21
image_size = [224, 224]
debug = 0

def get_box(keypoints, box_factor=1):
    box = []
    hand_min = np.min(keypoints, axis=0)
    hand_max = np.max(keypoints, axis=0)
    hand_box_c = (hand_max + hand_min) / 2.
    half_size = int(np.max(hand_max - hand_min) * box_factor / 2.)

    x_left = int(hand_box_c[0] - half_size)
    y_top = int(hand_box_c[1] - half_size)
    x_right = x_left + 2 * half_size
    y_bottom = y_top + 2 * half_size
    return box[(x_left, y_top), (x_right, y_bottom)]


def rotation(image_dir, keypoints, alignmenter):
    ori_joints = np.array(keypoints).reshape(21, 3)
    data_numpy = cv2.imread(image_dir)
    # data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)
    input_data, warp_matrix = alignmenter(data_numpy, ori_joints)
    return input_data


def init(txt_dir):
    open(txt_dir, 'w').close()
    if debug:
        cv2.namedWindow("aa", cv2.WINDOW_NORMAL)
    is_exists = os.path.exists(SAVE_PATH)
    if not is_exists:
        os.makedirs(SAVE_PATH)
        print('path of %s is build' % SAVE_PATH)

    json_file = make_json_head()
    hand_mode = HandLandModel(capability=1)
    alignmenter = HandInfo(img_size=image_size[0])
    return json_file, hand_mode, alignmenter

def main():
    json_head, hand_mode, alignmenter = init(txt_dir)
    goodcase_json = deepcopy(json_head)
    badcase_json = deepcopy(json_head)

    with open(json_dir, 'r')as f:
        json_data = json.load(f)
        images = json_data['images']
        annotations = json_data['annotations']
        assert (len(images) == len(annotations))

    count = 0
    for index in tqdm(range(len(images))):
        # if index < 1500:
        #     continue
        image_info = images[index]
        annotation_info = annotations[index]

        image_dir = image_info['image_dir']
        keypoints = annotation_info['keypoints']

        input_data = rotation(image_dir, keypoints, alignmenter)
        landmark, handness, righthand_prop, _ = hand_mode.run(input_data)

        if righthand_prop > 0.5:
            hand_type = 'right'
        else:
            hand_type = 'left'

        if debug:
            img = cv2.putText(input_data, f'{hand_type}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1)
            # img = draw_2d_points(landmark, deepcopy(img))
            cv2.imshow('aa', img)
            cv2.waitKey(1) & 0xFF


        if handness < 0.5:
            with open(txt_dir, 'a') as f:
                f.write(image_dir + '\n')
                print(f'No hands\t{image_dir}')
            convert_coco_format_left_label(badcase_json, deepcopy(image_info), deepcopy(annotation_info), hand_type, handness[0])

        else:
            convert_coco_format_left_label(goodcase_json, deepcopy(image_info), deepcopy(annotation_info), hand_type, handness[0])


    json_path = os.path.join(SAVE_PATH, JSON_NAME)
    with open(json_path, 'w') as fw:
        json.dump(goodcase_json, fw)
        print(f"{json_path} have succeed to write")

    json_path = os.path.join(SAVE_PATH, BADCASE_JSON_NAME)
    with open(json_path, 'w') as fw:
        json.dump(badcase_json, fw)
        print(f"{json_path} have succeed to write")


if __name__ == "__main__":
    main()