import argparse
import os
import numpy as np
from json_tools import _init_save_folder
import json
import cv2
import copy
from tqdm import tqdm
import json
from convert_tools import convert_coco_format_from_wholebody

COCOBBOX_FACTOR = 1.5
COCO_START_ID = 1_200_000
VAL_NUM = 1_500
Divide_1 = COCO_START_ID + VAL_NUM


def main(save_dir, data_dir, coco_start_id):
    # 创建文件结构并返回coco规范的dict结构
    json_file_model = _init_save_folder(save_dir)
    json_trainfile = copy.deepcopy(json_file_model)
    json_valfile = copy.deepcopy(json_file_model)
    json_testfile = copy.deepcopy(json_file_model)

    # 载入开始id
    coco_id = coco_start_id

    filenames = os.listdir(data_dir)
    for filename in filenames:
        if not os.path.isdir(os.path.join(data_dir, filename)) or filename not in ["val", "train"]:
            continue

        # sort_coco_wholebody_train_v1.0.json
        json_name = 'sort_coco_wholebody_' + filename + '_v1.0.json'
        json_dir = os.path.join(data_dir, json_name)

        with open(json_dir, 'r') as f:
            json_infos = json.load(f)
            img_infos = json_infos['images']
            anno_infos = json_infos['annotations']

            num_len = len(img_infos)

            for i in tqdm(range(num_len)):
                img_info = img_infos[i]
                anno_info = anno_infos[i]

                img_name = img_info["file_name"]
                img_path = os.path.join(data_dir, filename, img_name)

                hands_kpts = []
                if anno_info["lefthand_valid"]:
                    hands_kpts.append(anno_info["lefthand_kpts"])
                if anno_info["righthand_valid"]:
                    hands_kpts.append(anno_info["righthand_kpts"])

                for hand_kpts in hands_kpts:
                    flag = 0
                    kp = np.array(hand_kpts).reshape(21, 3)
                    kp[:, 2] = 2

                    if coco_id  < Divide_1:
                        mode = 'val2017'
                    else:
                        mode = 'train2017'

                    if type(kp) == np.ndarray:
                        if not os.path.exists(img_path) or np.all(kp[2]) == 0:
                            continue
                        img = cv2.imread(img_path)
                        if mode == "train2017":
                            flag = convert_coco_format_from_wholebody(img, img_path, kp, json_trainfile, mode, save_dir, coco_id)
                        if mode == "val2017":
                            flag = convert_coco_format_from_wholebody(img, img_path, kp, json_valfile, mode, save_dir, coco_id)
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
    parser.add_argument("--SavePath",
                        default=r"F:\image\COCO_whole_body\coco_from_whole_body",
                        help="this parameter is about the PATH of ImgSave", dest="SavePath", type=str)
    parser.add_argument("--DataPath",
                        default=r"F:\image\coco",
                        help="this parameter is about the PATH of ImgSave", dest="DataPath", type=str)
    # 解析参数
    args = parser.parse_args()
    print("parameter 'save_path' is :", args.SavePath)
    print("parameter 'data_path' is :", args.DataPath)
    main(args.SavePath, args.DataPath, coco_start_id=COCO_START_ID)