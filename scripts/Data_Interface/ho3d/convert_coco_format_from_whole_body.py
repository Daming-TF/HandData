import argparse
import sys
import numpy as np
import cv2
import copy
from tqdm import tqdm
import json
from random import shuffle

import os
DIR = os.path.split(os.path.realpath(__file__))[0]
DIR = os.path.split(DIR)[0]
sys.path.append(DIR)
from .vis_hand import canonical_coordinates, load_pickle_data, projectPoints, get_intrinsics
from library.json_tools import _init_save_folder
from convert_tools import convert_coco_format_from_wholebody

COCOBBOX_FACTOR = 1.5
COCO_START_ID = 1_300_000
VAL_NUM= 1_500
Divide_1 = COCO_START_ID + VAL_NUM
# END = Divide_2 + TEST_NUM

def main(save_dir, data_dir, coco_start_id):
    # 创建文件结构并返回coco规范的dict结构
    json_file_model = _init_save_folder(save_dir)
    json_trainfile = copy.deepcopy(json_file_model)
    json_valfile = copy.deepcopy(json_file_model)
    json_testfile = copy.deepcopy(json_file_model)

    # 载入开始id
    coco_id = coco_start_id

    print("loading the img path to the list")
    img_path_list = []
    filenames_1 = os.listdir(data_dir)
    for filename in filenames_1:
        if not os.path.isdir(os.path.join(data_dir, filename)) or filename not in ["train"]:
            continue
        # ./evaluation
        data_dir_1 = os.path.join(data_dir, filename)

        datanames = os.listdir(data_dir_1)
        for dataname in datanames:
            # ./evaluation/AP10/rgb
            data_dir_rgb = os.path.join(data_dir_1, dataname, "rgb")

            img_names = os.listdir(data_dir_rgb)
            for img_name in tqdm(img_names):
                if img_name.split(".")[-1] == "jpg":
                    # ./evaluation/AP10/rgb/0000.jpg
                    img_path = os.path.join(data_dir_rgb, img_name)
                    img_path_list.append(img_path)

            print(f"{dataname} is finished")

    shuffle(img_path_list)
    num_len = len(img_path_list)
    print(f"There are {num_len} data in total")

    # img  kp
    for i in tqdm(range(num_len)):
        img_dir = img_path_list[i]
        img_name = os.path.basename(img_dir)
        # print(img_dir)

        # ./evaluation/AP10
        path = img_dir[:img_dir.find("rgb")-1]
        # AP10
        data_name = os.path.basename(path)
        id = path[-1]
        inter_param_path = os.path.join(data_dir, "calibration", data_name[:-1],
                                   "calibration", 'cam_'+id+'_intrinsics.txt')
        # print(inter_param_path)
        pickle_path = os.path.join(path, "meta", img_name.split(".")[0] + ".pkl")
        # print(pickle_path)
        # get Camera internal parameters
        K = get_intrinsics(inter_param_path).tolist()
        # print(img_dir)
        optPickData = load_pickle_data(pickle_path)
        handJoints3D= optPickData['handJoints3D']
        if np.all(handJoints3D == 0) or np.all(handJoints3D == None):
            continue

        if not handJoints3D.shape == (21, 3):
            print(img_dir + " is lost")
            continue
        kp = projectPoints(handJoints3D, K)
        kp = canonical_coordinates(kp)

        if coco_id < Divide_1:
            mode = 'val2017'
        else:
            mode = 'train2017'

        if type(kp) == np.ndarray:
            if not os.path.exists(img_path) or np.all(kp[2]) == 0:
                continue
            img = cv2.flip(cv2.imread(img_dir), 1)
            if mode == "train2017":
                flag = convert_coco_format_from_wholebody(img, img_dir, kp, json_trainfile, mode, save_dir, coco_id)
            if mode == "val2017":
                flag = convert_coco_format_from_wholebody(img, img_dir, kp, json_valfile, mode, save_dir, coco_id)
            if flag == 1:
                coco_id += 1

    with open(os.path.join(save_dir, 'annotations', f'person_keypoints_val2017.json'), 'w') as fw:
        json.dump(json_valfile, fw)
        print("person_keypoints_val2017.json have succeed to write")
    with open(os.path.join(save_dir, 'annotations', f'person_keypoints_train2017.json'), 'w') as fw:
        json.dump(json_trainfile, fw)
        print("person_keypoints_train2017.json have succeed to write")


if __name__ == "__main__":
    # 创建一个解析器
    parser = argparse.ArgumentParser()
    # 添加参数ImgDatPath
    parser.add_argument("--SavePath",
                        default=r"G:\imgdate2\HO3D_v3\HO3D_from_whole_body",
                        help="this parameter is about the PATH of ImgSave", dest="SavePath", type=str)
    parser.add_argument("--DataPath",
                        default=r"G:\imgdate2\HO3D_v3\HO3D_v3",
                        help="this parameter is about the PATH of ImgSave", dest="DataPath", type=str)
    # 解析参数
    args = parser.parse_args()
    print("parameter 'save_path' is :", args.SavePath)
    print("parameter 'data_path' is :", args.DataPath)
    main(args.SavePath, args.DataPath, coco_start_id=COCO_START_ID)
