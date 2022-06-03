import argparse
import os
import numpy as np
from json_tools import _init_save_folder, convert_coco_format
from convert_tools import convert_coco_format_from_wholebody
import json
import cv2
import copy
from tqdm import tqdm

COCOBBOX_FACTOR = 1.5
COCO_START_ID = 1000_000
VAL_NUM = 1_500
Divide_1 = COCO_START_ID + VAL_NUM

def main(data_path, save_path):
    # 创建文件结构并返回coco规范的dict结构
    json_file_model = _init_save_folder(save_path)
    val_json, train_json = copy.deepcopy(json_file_model), copy.deepcopy(json_file_model)
    # 载入开始id
    coco_id = COCO_START_ID

    image_paths, json_paths = [], []
    files = os.listdir(data_path)
    for file in files:
        file_path = os.path.join(data_path, file)
        for keystring in ["train", "val", "test"]:
            if file.find(keystring) > 0:
                json_file = copy.deepcopy(json_file_model)

                filenames = os.listdir(file_path)       # 存放图片数据
                for filename in tqdm(filenames):
                    if filename.endswith(".jpg"):
                        img_path = os.path.join(file_path, filename)
                        json_path = os.path.join(file_path, filename[:filename.rfind(".")]+".json")
                        image_paths.append(img_path)
                        json_paths.append(json_path)

    data_num = len(image_paths)
    shuffle_list = np.arange(data_num)
    np.random.shuffle(shuffle_list)

    for i in tqdm(range(data_num)):
        index = shuffle_list[i]
        json_path = json_paths[index]
        img_path = image_paths[index]

        # 加载非coco格式的json文件
        hand_pts = json.load(open(json_path))
        if type(hand_pts['hand_pts']) == list:
            img = cv2.imread(img_path)

            kp = np.array(hand_pts['hand_pts'])
            # img, img_path, landmarks, json_file, mode, save_dir, img_id

            if coco_id >= COCO_START_ID and coco_id < Divide_1:
                mode = 'val2017'
                json_file = val_json
            elif coco_id >= Divide_1:
                mode = 'train2017'
                json_file = train_json

            flag = convert_coco_for_mat_from_wholebody(img_path, kp, json_file, mode, save_path, coco_id)
            if flag == 1:
                coco_id +=1
                                # print(coco_id)

    with open(os.path.join(save_path, 'annotations', f'person_keypoints_val.json'), 'w') as fw:
        json.dump(val_json, fw)
    with open(os.path.join(save_path, 'annotations', f'person_keypoints_train.json'), 'w') as fw:
        json.dump(train_json, fw)

    print(
f'''

————We are standardizing the json file format, 
    and now the json file preparation of the >> {file} << file has been completed
    

'''
                )






if __name__ == "__main__":
    # 创建一个解析器
    parser = argparse.ArgumentParser()
    # 添加参数ImgDatPath
    parser.add_argument("--DataPath",
                        default=r"F:\image\CMU\hand_labels_synth\hand_labels_synth",
                        help="this parameter is about the PATH of ImgSave", dest="DataPath", type=str)
    parser.add_argument("--SavePath",
                        default=r"F:\image\CMU\hand_labels_synth\hand_labels_synth_from_whole_body",
                        help="this parameter is about the PATH of ImgSave", dest="SavePath", type=str)
    # 解析参数
    args = parser.parse_args()
    print("parameter 'save_path' is :", args.SavePath)
    main(args.DataPath, args.SavePath)