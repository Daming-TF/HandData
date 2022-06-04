"""
    该程序主要是为了把标注的Json格式转成我们的COCO-Json格式
"""
import argparse
import copy
import json

from tqdm import tqdm
import os
import cv2
import numpy as np

from library.json_tools import _init_save_folder
from Tag_tools import convert_coco_format

COCOBBOX_FACTOR = 1.5
COCO_START_ID = 1_500_000
VAL_NUM = 1_500

Divide = COCO_START_ID + VAL_NUM


def set_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--SavePath", help='生成的crop-image和注释信息保存位置',
                        default=r"G:\test_data\hardcase_data\hardcase")
    parser.add_argument("--DataPath", help='hardcase数据集原图路径',
                        default=r"G:\test_data\hardcase_data\dataset")
    args = parser.parse_args()
    return args

member_list = ['chengbin', 'jiahui', 'aiyu', 'huawei', 'lihui', 'yangyun', 'wangfaqiang',
               'linzhe', 'xintong','hongyu','zhiwen', 'haoye', 'wansen']
def main():
    coco_id = COCO_START_ID

    args = set_parser()
    save_path = args.SavePath
    data_path = args.DataPath

    json_head = _init_save_folder(save_path)
    val_json_head, train_json_head = copy.deepcopy(json_head), copy.deepcopy(json_head)

    image_dirs, json_dirs = [], []
    for team_member_name in member_list:
        data_dir = os.path.join(data_path, team_member_name)
        image_path = os.path.join(data_dir, 'images')
        image_names = os.listdir(image_path)
        for image_name in image_names:
            if image_name.endswith('.jpg'):
                img_dir = os.path.join(data_dir, 'images', image_name)
                json_dir = os.path.join(data_dir, 'anno', image_name.split('.')[0]+'.json')
                image_dirs.append(img_dir)
                json_dirs.append(json_dir)

    indexs = np.arange(len(image_dirs))
    np.random.shuffle(indexs)

    for i in tqdm(range(indexs.shape[0])):
        index = indexs[i]
        img_dir = image_dirs[index]
        json_dir = json_dirs[index]
        with open(json_dir, 'r')as f:
            json_data = json.load(f)
            keypoints_list = json_data['annotations'][0]['keypoints']
            if keypoints_list == []:
                continue
            for keypoints in keypoints_list:
                kp = np.zeros([21, 3])
                keypoints = np.array(keypoints).reshape(21, 2)
                kp[:, :2] = keypoints
                kp[:, 2] = 2

                if coco_id < Divide:
                    head = val_json_head
                    mode = 'val2017'

                else:
                    head = train_json_head
                    mode = 'train2017'

                flag = convert_coco_format(img_dir, kp, head, mode, save_path, coco_id)
                if flag == 1:
                    coco_id += 1

    save_dir = os.path.join(save_path, 'annotations', 'train2017.json')
    with open(save_dir, 'w') as fw:
        json.dump(train_json_head, fw)
        print("train2017.json have succeed to write")

    save_dir = os.path.join(save_path, 'annotations', 'val2017.json')
    with open(save_dir, 'w') as fw:
        json.dump(val_json_head, fw)
        print("val2017.json have succeed to write")


if __name__ == "__main__":
    main()
