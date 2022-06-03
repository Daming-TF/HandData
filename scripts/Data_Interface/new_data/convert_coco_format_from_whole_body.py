import argparse
import os
import numpy as np
from json_tools import _init_save_folder
from convert_tools import convert_coco_format_from_wholebody
import json
import cv2
import copy
from tqdm import tqdm

COCOBBOX_FACTOR = 1.5
COCO_START_ID = 1_400_000
# VAL_NUM = 1_500

# Divide_1 = COCO_START_ID + VAL_NUM

def main(data_path, save_dir, json_dir, coco_start_id):
    # 创建文件结构并返回coco规范的dict结构
    json_file_model = _init_save_folder(save_dir)

    # 载入开始id
    coco_id = coco_start_id
    with open(json_dir, "r") as f:
        json_infos = json.load(f)
        images = json_infos['images']
        annotations = json_infos['annotations']

    data_num = len(images)
    # json_trainfile = copy.deepcopy(json_file_model)
    # json_valfile = copy.deepcopy(json_file_model)

    # 乱序
    # shuffle_list = np.arange(data_num)
    # np.random.shuffle(shuffle_list)

    for i in tqdm(range(data_num)):
        # i = shuffle_list[index]
        image_info = images[i]
        annotation_info = annotations[i]
        img_path = image_info['image_dir']
        kp = np.array(annotation_info['keypoints']).reshape(21, 3)

        mode = 'test2017'
        flag = convert_coco_format_from_wholebody(img_path, kp, json_file_model, mode, save_dir, coco_id)

        if flag == 1:
            coco_id += 1

    with open(os.path.join(save_dir, 'annotations', f'person_keypoints_test2017.json'), 'w') as fw:
        json.dump(json_file_model, fw)
        print("person_keypoints_train2017.json have succeed to write")


if __name__ == "__main__":
    # 创建一个解析器
    parser = argparse.ArgumentParser()
    # 添加参数ImgDatPath
    parser.add_argument("--DataPath",
                        default=r"G:\test_data\new_data\dataset",
                        help="this parameter is about the PATH of ImgSave", dest="DataPath", type=str)
    parser.add_argument("--SavePath",
                        default=r"G:\test_data\new_data\new_data_from_whole_body",
                        help="this parameter is about the PATH of ImgSave", dest="SavePath", type=str)
    parser.add_argument("--JsonPath",
                        default=r"G:\test_data\new_data\new_data_from_whole_body\total.json",
                        help="this parameter is about the PATH of ImgSave", dest="JsonPath", type=str)
    # 解析参数
    args = parser.parse_args()
    print("parameter 'save_path' is :", args.SavePath)
    print("parameter 'json_path' is :", args.JsonPath)
    main(args.DataPath, args.SavePath, args.JsonPath, coco_start_id=COCO_START_ID)