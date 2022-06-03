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
COCO_START_ID = 1_200_000
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
                                                      image_id, hand_type=hand_type, save_flag=flag)
            if Debug:
                debug_image = draw_2d_points(keypoints, debug_image)
                txt = 'l' if hand_type == 'left' else 'r'
                debug_image = cv2.putText(debug_image, txt, (int(keypoints[0, 0]), int(keypoints[0, 1])),
                                    cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 2)

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

    anno_dict = defaultdict(list)
    img_dict = dict()

    filenames = os.listdir(data_dir)
    for filename in filenames:
        if not os.path.isdir(os.path.join(data_dir, filename)) or filename not in ["train", "val"]:      # "train",
            continue

        # sort_coco_wholebody_train_v1.0.json
        json_name = 'coco_wholebody_' + filename + '_v1.0.json'
        json_dir = os.path.join(data_dir, json_name)

        with open(json_dir, 'r') as f:
            json_infos = json.load(f)
            img_infos = json_infos['images']
            anno_infos = json_infos['annotations']

        print("loading images ......")
        for img_info in tqdm(img_infos):
            img_dict[img_info['id']] = img_info

        print("loading annotations ......")
        for anno_info in tqdm(anno_infos):
            anno_dict[anno_info['image_id']].append(anno_info)

        for image_id in anno_dict.keys():
            anno_list = anno_dict[image_id]
            img = img_dict[image_id]

            # extract image path
            img_name = img["file_name"]
            img_path = os.path.join(data_dir, filename, img_name)
            if not os.path.exists(img_path):
                continue

            for anno in anno_list:
                # extract keypoints
                hands = {}
                if anno["lefthand_valid"]:
                    hands['left'] = np.array(anno["lefthand_kpts"]).reshape(21, 3)
                if anno["righthand_valid"]:
                    hands['right'] = np.array(anno["righthand_kpts"]).reshape(21, 3)

                for hand_type in hands.keys():
                    keypoints = hands[hand_type]

                    unit_dict = dict({
                        'image_path': img_path,
                        'keypoints': keypoints,
                        'hand_type': hand_type,
                    })

                    tag_name = f'{filename}_{img_name}'
                    data_dict[tag_name].append(unit_dict)

    return data_dict, list(data_dict.keys())


if __name__ == "__main__":
    # 创建一个解析器
    parser = argparse.ArgumentParser()
    # 添加参数ImgDatPath
    parser.add_argument("--SaveDir", default=r"F:\image\COCO_whole_body\coco_from_whole_body_v2_6")
    parser.add_argument("--DataDir", default=r"F:\image\coco")
    # 解析参数
    args = parser.parse_args()
    main(args.DataDir, args.SaveDir)