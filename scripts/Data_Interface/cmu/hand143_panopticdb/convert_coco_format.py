import argparse
import os
import numpy as np
import json
import cv2
import copy
from tqdm import tqdm

from library.json_tools import _init_save_folder, convert_coco_format

COCOBBOX_FACTOR = 1.5
COCO_START_ID = 900_000
TRAIN_NUM, VAL_NUM, TEST_NUM = 11_817, 1_500, 1_500

Divide_1 = COCO_START_ID + TRAIN_NUM
Divide_2 = Divide_1 + VAL_NUM
END = Divide_2 + TEST_NUM

def main(save_dir, json_dir, coco_start_id):
    # 创建文件结构并返回coco规范的dict结构
    json_file_model = _init_save_folder(save_dir)
    # 载入开始id
    coco_id = coco_start_id
    with open(json_dir, "r") as f:
        json_infos = json.load(f)
        hands = json_infos['root']
        nums_hands = len(hands)
        json_trainfile = copy.deepcopy(json_file_model)
        json_valfile = copy.deepcopy(json_file_model)
        json_testfile = copy.deepcopy(json_file_model)
        for i in tqdm(range(nums_hands)):
            hand_pts = hands[i]['joint_self']
            img_path = os.path.join(save_dir, hands[i]['img_paths'].strip(""))

            if coco_id >= COCO_START_ID and coco_id < Divide_1:
                mode = 'train2017'
            elif coco_id >= Divide_1 and coco_id < Divide_2:
                mode = 'val2017'
            elif coco_id >= Divide_2 and coco_id < END:
                mode = 'test2017'

            if type(hand_pts) == list:
                # print(img_path)
                img = cv2.imread(img_path)
                kp = np.array(hand_pts)
                if mode == "train2017":
                    flag = convert_coco_format(img, kp, json_trainfile, mode, save_dir, coco_id)
                if mode == "val2017":
                    flag = convert_coco_format(img, kp, json_valfile, mode, save_dir, coco_id)
                if mode == "test2017":
                    flag = convert_coco_format(img, kp, json_testfile, mode, save_dir, coco_id)
                if flag == 1:
                    coco_id += 1

        with open(os.path.join(save_dir, 'annotations', f'person_keypoints_train2017.json'), 'w') as fw:
            json.dump(json_trainfile, fw)
            print("person_keypoints_train2017.json have succeed to write")
        with open(os.path.join(save_dir, 'annotations', f'person_keypoints_val2017.json'), 'w') as fw:
            json.dump(json_valfile, fw)
            print("person_keypoints_val2017.json have succeed to write")
        with open(os.path.join(save_dir, 'annotations', f'person_keypoints_test2017.json'), 'w') as fw:
            json.dump(json_testfile, fw)
            print("person_keypoints_test2017.json have succeed to write")


if __name__ == "__main__":
    # 创建一个解析器
    parser = argparse.ArgumentParser()
    # 添加参数ImgDatPath
    parser.add_argument("--SavePath",
                        default=r"F:\image\CMU\hand143_panopticdb\hand143_panopticdb",
                        help="this parameter is about the PATH of ImgSave", dest="SavePath", type=str)
    parser.add_argument("--JsonPath",
                        default=r"F:\image\CMU\hand143_panopticdb\hand143_panopticdb\hands_v143_14817.json",
                        help="this parameter is about the PATH of ImgSave", dest="JsonPath", type=str)
    # 解析参数
    args = parser.parse_args()
    print("parameter 'save_path' is :", args.SavePath)
    print("parameter 'json_path' is :", args.JsonPath)
    main(args.SavePath, args.JsonPath, coco_start_id=COCO_START_ID)