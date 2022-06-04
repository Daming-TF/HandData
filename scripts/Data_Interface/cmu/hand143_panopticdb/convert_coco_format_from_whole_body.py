import argparse
import os
import numpy as np
import json
import copy
from tqdm import tqdm

from library.json_tools import _init_save_folder
from convert_tools import convert_coco_format_from_wholebody

COCOBBOX_FACTOR = 1.5
COCO_START_ID = 900_000
VAL_NUM = 1_500

Divide_1 = COCO_START_ID + VAL_NUM

def main(data_path, save_dir, json_dir, coco_start_id):
    # 创建文件结构并返回coco规范的dict结构
    json_file_model = _init_save_folder(save_dir)
    # 载入开始id
    coco_id = coco_start_id
    with open(json_dir, "r") as f:
        json_infos = json.load(f)
        hands = json_infos['root']
    data_num = len(hands)
    json_trainfile = copy.deepcopy(json_file_model)
    json_valfile = copy.deepcopy(json_file_model)

    shuffle_list = np.arange(data_num)
    np.random.shuffle(shuffle_list)

    for index in tqdm(range(data_num)):
        i = shuffle_list[index]
        hand_pts = hands[i]['joint_self']
        img_path = os.path.join(data_path, hands[i]['img_paths'].strip(""))

        if coco_id >= COCO_START_ID and coco_id < Divide_1:
            mode = 'val2017'
        elif coco_id >= Divide_1:
            mode = 'train2017'

        if type(hand_pts) == list:
            # print(img_path)
            kp = np.array(hand_pts)
            if mode == "train2017":
                flag = convert_coco_format_from_wholebody(img_path, kp, json_trainfile, mode, save_dir, coco_id)
            if mode == "val2017":
                flag = convert_coco_format_from_wholebody(img_path, kp, json_valfile, mode, save_dir, coco_id)
            if flag == 1:
                coco_id += 1
    with open(os.path.join(save_dir, 'annotations', f'person_keypoints_train2017.json'), 'w') as fw:
        json.dump(json_trainfile, fw)
        print("person_keypoints_train2017.json have succeed to write")
    with open(os.path.join(save_dir, 'annotations', f'person_keypoints_val2017.json'), 'w') as fw:
        json.dump(json_valfile, fw)
        print("person_keypoints_val2017.json have succeed to write")


if __name__ == "__main__":
    # 创建一个解析器
    parser = argparse.ArgumentParser()
    # 添加参数ImgDatPath
    parser.add_argument("--DataPath",
                        default=r"F:\image\CMU\hand143_panopticdb\hand143_panopticdb",
                        help="this parameter is about the PATH of ImgSave", dest="DataPath", type=str)
    parser.add_argument("--SavePath",
                        default=r"F:\image\CMU\hand143_panopticdb\hand143_pannopticdb_from_whole_body",
                        help="this parameter is about the PATH of ImgSave", dest="SavePath", type=str)
    parser.add_argument("--JsonPath",
                        default=r"F:\image\CMU\hand143_panopticdb\hand143_panopticdb\hands_v143_14817.json",
                        help="this parameter is about the PATH of ImgSave", dest="JsonPath", type=str)
    # 解析参数
    args = parser.parse_args()
    print("parameter 'save_path' is :", args.SavePath)
    print("parameter 'json_path' is :", args.JsonPath)
    main(args.DataPath, args.SavePath, args.JsonPath, coco_start_id=COCO_START_ID)