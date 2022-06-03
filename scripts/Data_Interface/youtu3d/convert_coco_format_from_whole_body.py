"""
    该程序主要是为了把标注的Json格式转成我们的COCO-Json格式
"""
import argparse
import copy
import json
from json_tools import _init_save_folder
from convert_tools import convert_coco_format_from_wholebody
from tqdm import tqdm
import os
import cv2
import numpy as np



COCOBBOX_FACTOR = 1.5
COCO_START_ID = 300_000
VAL_NUM = 1_500

Divide = COCO_START_ID + VAL_NUM


def set_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--SavePath", default=r"E:\Data\landmarks\YouTube3D\YouTube3D-wholebody")
    parser.add_argument("--DataPath", default=r"E:\Data\landmarks\YouTube3D\images")
    parser.add_argument("--JsonPath", default=r"E:\数据标记反馈\youtu3d\批次数据")
    args = parser.parse_args()
    return args


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
        handlandmarks_list.append(hand1)
    if not np.all(hand2 == 0):
        hand2[:, 2] = 2
        handlandmarks_list.append(hand2)
    return handlandmarks_list


def main():
    En_flag = 0
    args = set_parser()
    save_path = args.SavePath
    data_path = args.DataPath
    json_path = args.JsonPath

    json_head = _init_save_folder(save_path)
    train_json_head, val_json_head = copy.deepcopy(json_head), copy.deepcopy(json_head)

    originalFileName_list, labelFeature_list = [], []
    json_files = os.listdir(json_path)

    for json_file in json_files:
        json_dir = os.path.join(json_path, json_file)

        if json_file == '5053-手势关键点-2022_1_14.json':
            with open(json_dir, 'r', encoding='UTF-8') as f:
                json_data = json.load(f)
                label = json_data[0]['label']
        else:
            with open(json_dir, 'r', encoding='UTF-8') as f:
                json_data = json.load(f)
                label = json_data

        print(f"loading ({json_dir}) ......")
        for index in tqdm(range(len(label))):
            label_dect = label[index]
            originalFileName = label_dect['originalFileName']
            labelFeature = label_dect['labelFeature']
            originalFileName_list.append(originalFileName)
            labelFeature_list.append(labelFeature)
        print(f"Finish loaded 》{json_dir}《 !")

    assert (len(originalFileName_list) == len(labelFeature_list))
    indexs = np.arange(len(labelFeature_list))

    np.random.shuffle(indexs)

    coco_id = COCO_START_ID
    image_id = coco_id
    print("start to work ......")
    for i in tqdm(range(indexs.shape[0])):
        flag = 0
        index = indexs[i]
        originalFileName = originalFileName_list[index]
        labelFeature = labelFeature_list[index]

        image_name_list = originalFileName.split('_')
        image_name = image_name_list[1]
        image_dir = os.path.join(data_path, image_name_list[0] + '2017', image_name)
        # img = cv2.imread(image_dir)

        handlandmarks_list = get_keypoints(labelFeature)

        if len(handlandmarks_list) == 0:
            continue

        for handlandmarks in handlandmarks_list:
            if image_id < Divide:
                head = val_json_head
                mode = 'val'
                # En_flag = 1

            else:
                if image_id == Divide:
                    save_dir = os.path.join(save_path, 'annotations', 'val2017.json')
                    with open(save_dir, 'w') as fw:
                        json.dump(val_json_head, fw)
                        print("val2017.json have succeed to write")
                head = train_json_head
                mode = 'train'

            flag = convert_coco_format_from_wholebody(image_id, image_dir, handlandmarks, head, coco_id, save_path, mode)

            if flag == 1:
                coco_id += 1

        if flag == 1:
            image_id += 1

    save_dir = os.path.join(save_path, 'annotations', 'train2017.json')
    with open(save_dir, 'w') as fw:
        json.dump(train_json_head, fw)
        print("train2017.json have succeed to write")


if __name__ == "__main__":
    main()