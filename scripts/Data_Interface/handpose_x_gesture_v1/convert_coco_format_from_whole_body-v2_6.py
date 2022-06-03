"""
重新构建v2_6的数据集，一张图片对应一个images的一个unit，每个手对应annotations的一个unit
images['id']标识图片id
annotations['id']标识手对象id
annotation['image_id']标识图片id

同时把所有关键点坐标保留两位保存

对于不同数据集的convert_coco_format仅需要对load_data做修改
对于load_data: 参照coco的读取数据方式，以原始图片名做索引号，对每张图片存储关键信息单元如下
unit_dict = dict({
                'hand_type': hand_type,
                'image_path': img_path,
                'keypoints': keypoints,
            })

YouTu 3D数据集原始数据全部都给了标注团队标注，所以对于该数据集主要输入数据是标注团队反馈回来的json文件

"""
import argparse
import os
from tqdm import tqdm
import cv2
import numpy as np
import json
import copy
from collections import defaultdict
import time

from json_tools import _init_save_folder
from dataset_v2_6 import convert_coco_format_from_wholebody
from tools import draw_2d_points

COCOBBOX_FACTOR = 1.5
COCO_START_ID = 500_000
VAL_NUM = 0
Divide_1 = COCO_START_ID + VAL_NUM
Debug = 0


def main(data_dir, save_dir):
    # init
    json_file_model = _init_save_folder(save_dir)
    val_json, train_json = copy.deepcopy(json_file_model), copy.deepcopy(json_file_model)
    coco_id, image_id = COCO_START_ID, COCO_START_ID
    image_counter = 0

    # Load information using coco method
    print("loading the data......")
    data_dict, tag_names = load_data(data_dir)

    img_num = len(tag_names)
    shuffle_list = np.arange(img_num)
    rng = np.random.default_rng(12345)
    rng.shuffle(shuffle_list)

    print("processing data......")
    for index in tqdm(range(img_num)):
        shuffle_id = shuffle_list[index]
        tag_name = tag_names[shuffle_id]

        unit_list = data_dict[tag_name]
        if Debug:
            debug_image = cv2.imread(unit_list[0]['image_path'])

        flag = 0
        for hand_index, unit_dict in enumerate(unit_list):
            img_path = unit_dict['image_path']
            keypoints = unit_dict['keypoints']

            if VAL_NUM > image_counter >= 0:
                mode = 'val2017'
                json_file = val_json
            elif image_counter >= VAL_NUM:
                if image_counter == VAL_NUM and VAL_NUM != 0:
                    with open(os.path.join(save_dir, 'annotations', f'person_keypoints_val2017.json'), 'w') as fw:
                        json.dump(val_json, fw)
                mode = 'train2017'
                json_file = train_json

            flag = convert_coco_format_from_wholebody(img_path, keypoints, json_file, mode, save_dir, coco_id,
                                                      image_id, save_flag=flag)
            if Debug:
                debug_image = draw_2d_points(keypoints, debug_image)

            if flag == 1:
                coco_id +=1

        if Debug:
            cv2.imshow('show', debug_image)
            cv2.waitKey(0)

        image_id += 1
        if flag == 1:
            image_counter += 1

    print(f"Writing the json file person_keypoints_train.json......")
    with open(os.path.join(save_dir, 'annotations', f'person_keypoints_train2017.json'), 'w') as fw:
        json.dump(train_json, fw)


def load_data(data_dir):
    data_dict = defaultdict(list)

    json_paths = list()
    sub_folders = sorted(os.path.join(data_dir, file) for file in os.listdir(data_dir)
                         if not (file.endswith('.txt') or file == 'badcase'))

    for sub_folder in sub_folders:
        json_paths.extend(sorted(os.path.join(sub_folder, file) for file in os.listdir(sub_folder)
                                 if file.endswith('.json')))

    for index, json_path in tqdm(enumerate(json_paths)):
        if index == 2850:
            print()
        # extract image path
        image_path = json_path.replace('.json', '.jpg')

        # extract keypoints
        anno_infos = json.load(open(json_path, "r"))
        if 'error_msg' not in anno_infos:
            num_hands = len(anno_infos["hand_info"])
        else:
            continue

        if num_hands != 1:
            continue

        anno_info = anno_infos["hand_info"][0]
        hand_parts = anno_info["hand_parts"]
        keypoints = convert_joints(hand_parts)

        unit_dict = dict({
            'image_path': image_path,
            'keypoints': keypoints,
        })

        tag_name = os.path.basename(image_path)
        data_dict[tag_name].append(unit_dict)

    return data_dict, list(data_dict.keys())


def convert_joints(hand_parts):
    joints = np.zeros((21, 3), dtype=float)
    for i in range(21):
        joints[i, 0] = hand_parts[str(i)]['x']
        joints[i, 1] = hand_parts[str(i)]['y']
        joints[i, 2] = hand_parts[str(i)]['score']
    return joints


if __name__ == "__main__":
    # 创建一个解析器
    parser = argparse.ArgumentParser()
    # 添加参数ImgDatPath
    parser.add_argument("--SaveDir", default=r"E:\Data\landmarks\handpose_x_gesture_v1\HXG_from_whole_body_v2_6")
    parser.add_argument("--DataDir", default=r"E:\Data\landmarks\handpose_x_gesture_v1\handpose_x_gesture_v1")
    # 解析参数
    args = parser.parse_args()
    main(args.DataDir, args.SaveDir)