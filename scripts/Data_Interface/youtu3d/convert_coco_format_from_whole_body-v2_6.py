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

from library.json_tools import _init_save_folder
from library.dataset_v2_6 import convert_coco_format_from_wholebody
from library.tools import draw_2d_points

COCOBBOX_FACTOR = 1.5
COCO_START_ID = 300_000
VAL_NUM = 1500
Divide_1 = COCO_START_ID + VAL_NUM
Debug = 0


def main(data_dir, json_dir, save_dir):
    # init
    json_file_model = _init_save_folder(save_dir)
    val_json, train_json = copy.deepcopy(json_file_model), copy.deepcopy(json_file_model)
    coco_id, image_id = COCO_START_ID, COCO_START_ID
    image_counter = 0

    # Load information using coco method
    print("loading the data......")
    data_dict, tag_names = load_data(data_dir, json_dir)

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
                if image_counter == VAL_NUM:
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


def load_data(data_dir, json_dir):
    data_dict = defaultdict(list)

    json_files = os.listdir(json_dir)
    for json_file in json_files:
        json_path = os.path.join(json_dir, json_file)

        if json_file == '5053-手势关键点-2022_1_14.json':
            with open(json_path, 'r', encoding='UTF-8') as f:
                json_data = json.load(f)
                label = json_data[0]['label']
        else:
            with open(json_path, 'r', encoding='UTF-8') as f:
                json_data = json.load(f)
                label = json_data

        for index in tqdm(range(len(label))):
            label_dect = label[index]
            originalFileName = label_dect['originalFileName']
            labelFeature = label_dect['labelFeature']

            # extract the image path
            image_name_list = originalFileName.split('_')
            file_name = image_name_list[0]
            image_name = image_name_list[1]
            image_path = os.path.join(data_dir, file_name + '2017', image_name)

            tag_name = f'{file_name}_{image_name}'

            # extract the keypoints
            keypoints_list = get_keypoints(labelFeature)
            for keypoints in keypoints_list:
                unit_dict = dict({
                    'image_path': image_path,
                    'keypoints': keypoints,
                })

                data_dict[tag_name].append(unit_dict)

    return data_dict, list(data_dict.keys())


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
    kps_valid_bool = keypoints[:, -1].astype(bool)
    key_pts = keypoints[:, :2][kps_valid_bool]
    if np.shape(key_pts)[0] >= 16:
        return 1
    else:
        return 0


if __name__ == "__main__":
    # 创建一个解析器
    parser = argparse.ArgumentParser()
    # 添加参数ImgDatPath
    parser.add_argument("--SaveDir", default=r"E:\Data\landmarks\YouTube3D\YouTube3D_from_whole_body_v2_6")
    parser.add_argument("--DataDir", default=r"E:\Data\landmarks\YouTube3D\images")
    parser.add_argument("--JsonDir", default=r"E:\数据标记反馈\youtu3d\批次数据")
    # 解析参数
    args = parser.parse_args()
    main(args.DataDir, args.JsonDir, args.SaveDir)
