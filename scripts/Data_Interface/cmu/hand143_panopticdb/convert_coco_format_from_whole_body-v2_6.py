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
"""
import argparse
import os
from tqdm import tqdm
import cv2
import numpy as np
import json
import copy
from collections import defaultdict

from library.json_tools import _init_save_folder
from library.dataset_v2_6 import convert_coco_format_from_wholebody
from library.tools import draw_2d_points

COCOBBOX_FACTOR = 1.5
COCO_START_ID = 900_000
VAL_NUM = 0
Divide_1 = COCO_START_ID + VAL_NUM
Debug = 0


def main(data_dir, json_path, save_dir):
    # init
    json_file_model = _init_save_folder(save_dir)
    val_json, train_json = copy.deepcopy(json_file_model), copy.deepcopy(json_file_model)
    coco_id, image_id = COCO_START_ID, COCO_START_ID
    image_counter = 1

    # Load information using coco method
    print("loading the data......")
    data_dict, tag_names = load_data(data_dir, json_path)

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
                if image_counter == VAL_NUM:
                    with open(os.path.join(save_dir, 'annotations', f'person_keypoints_val2017.json'), 'w') as fw:
                        json.dump(val_json, fw)
                mode = 'train2017'
                json_file = train_json

            flag = convert_coco_format_from_wholebody(img_path, keypoints, json_file, mode, save_dir, coco_id,
                                                      image_id, hand_type=hand_type, save_flag=flag)
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


def load_data(data_dir, json_path):
    data_dict = defaultdict(list)

    with open(json_path, 'r')as f:
        json_data = json.load(f)['root']

    for data_unit in json_data:
        keypoints = np.array(data_unit['joint_self']).reshape(21, 3)
        file_name, image_name = data_unit['img_paths'].split('/')
        img_path = os.path.join(data_dir, file_name, image_name)
        image_id = os.path.basename(img_path).split('.jpg')[0]

        unit_dict = dict({
            'image_path': img_path,
            'keypoints': keypoints,
            'hand_type': 'right'
        })
        data_dict[image_id].append(unit_dict)

    print("\n")
    return data_dict, list(data_dict.keys())


if __name__ == "__main__":
    # 创建一个解析器
    parser = argparse.ArgumentParser()
    # 添加参数ImgDatPath
    parser.add_argument("--DataDir",
                        default=r"F:\image\CMU\hand143_panopticdb\hand143_panopticdb",
                        help="this parameter is about the PATH of ImgSave", type=str)
    parser.add_argument("--JsonPath",
                        default=r"F:\image\CMU\hand143_panopticdb\hand143_panopticdb\hands_v143_14817.json",
                        help="this parameter is about the PATH of ImgSave", type=str)
    parser.add_argument("--SaveDir",
                        default=r"F:\image\CMU\hand143_panopticdb\hand143_panopticdb_from_whole_body_v2_6",
                        help="this parameter is about the PATH of ImgSave", type=str)
    # 解析参数
    args = parser.parse_args()
    main(args.DataDir, args.JsonPath, args.SaveDir)