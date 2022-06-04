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

from library.json_tools import _init_save_folder
from library.dataset_v2_6 import convert_coco_format_from_wholebody
from library.tools import draw_2d_points

COCOBBOX_FACTOR = 1.5
COCO_START_ID = 400_000
VAL_NUM = 0
Divide_1 = COCO_START_ID + VAL_NUM
Debug = 0

NUM_BODY_KEYPOINTS = 26
NUM_FACE_KEYPOINTS = 68
NUM_HAND_KEYPOINTS = 21


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
            hand_type = unit_dict['hand_type']

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
                                                      image_id, hand_type=hand_type ,save_flag=flag)
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


def load_data(data_dir, versions=['gs', 'hom', 'sample', 'auto']):
    data_dict = defaultdict(list)

    # init
    image_dir_list = [
        r'E:\Data\landmarks\HFB\halpe_Full-Body_Human_Keypoints_and_HOL-Det_dataset\hico_20160224_det\images\train2015',
        r'E:\Data\landmarks\HFB\halpe_Full-Body_Human_Keypoints_and_HOL-Det_dataset\val2017'
    ]
    keyword_list = ['train', 'val']

    file_names = os.listdir(data_dir)
    for file_name in file_names:
        if not file_name.endswith('.json'):
            continue

        for mode, image_dir in zip(keyword_list, image_dir_list):
            if mode in file_name:
                break

        json_path = os.path.join(data_dir, file_name)
        with open(json_path, 'r')as f:
            json_data = json.load(f)
            images = json_data['images']
            annotations = json_data['annotations']

        imgs = {}
        for image_unit in images:
            imgs[image_unit['id']] = image_unit

        for hidx, annot in enumerate(annotations):
            if 'keypoints' in annot and type(annot['keypoints']) == list:
                image_name = str(imgs[annot['image_id']]['file_name'])
                image_path = os.path.join(image_dir, image_name)

                kp = np.array(annot['keypoints'])
                kp_x = np.asarray(kp[0::3][NUM_BODY_KEYPOINTS + NUM_FACE_KEYPOINTS:]).reshape(2, NUM_HAND_KEYPOINTS)
                kp_y = np.asarray(kp[1::3][NUM_BODY_KEYPOINTS + NUM_FACE_KEYPOINTS:]).reshape(2, NUM_HAND_KEYPOINTS)
                kp_scores = np.asarray(kp[2::3][NUM_BODY_KEYPOINTS + NUM_FACE_KEYPOINTS:]).reshape(2,
                                                                                                   NUM_HAND_KEYPOINTS)

                r_hand = get_hand_keypoints(1, kp_x, kp_y, kp_scores)  # right
                l_hand = get_hand_keypoints(0, kp_x, kp_y, kp_scores)  # left

                # Draw keypoints
                for hand_type, keypoints in zip(['left', 'right'], [l_hand, r_hand]):
                    if np.sum(keypoints[:, 2]) >= NUM_HAND_KEYPOINTS:

                        unit_dict = dict({
                            'image_path': image_path,
                            'keypoints': keypoints,
                            'hand_type': hand_type
                        })

                        tag_name = f'{mode}_{image_name}'
                        data_dict[tag_name].append(unit_dict)

    return data_dict, list(data_dict.keys())


def get_hand_keypoints(hand_type, kp_x, kp_y, kp_scores):
    hand = np.zeros((NUM_HAND_KEYPOINTS, 3), dtype=float)
    if hand_type == 0:  # right
        hand[:, 0] = kp_x[0, :].copy()
        hand[:, 1] = kp_y[0, :].copy()
        hand[:, 2] = kp_scores[0, :].copy()
    else:   # left
        hand[:, 0] = kp_x[1, :].copy()
        hand[:, 1] = kp_y[1, :].copy()
        hand[:, 2] = kp_scores[1, :].copy()
    return hand


if __name__ == "__main__":
    # 创建一个解析器
    parser = argparse.ArgumentParser()
    # 添加参数ImgDatPath
    parser.add_argument("--SaveDir", default=r"E:\Data\landmarks\HFB\HFB_from_whole_body_v2_6")
    parser.add_argument("--DataDir", default=r"E:\Data\landmarks\HFB\halpe_Full-Body_Human_Keypoints_and_HOL-Det_dataset")
    # 解析参数
    args = parser.parse_args()
    main(args.DataDir, args.SaveDir)