import argparse
import os
import numpy as np
from json_tools import _init_save_folder, convert_coco_format
import json
import cv2
import copy
from tqdm import tqdm

COCOBBOX_FACTOR = 1.5

def main(save_dir, coco_start_id):
    # 创建文件结构并返回coco规范的dict结构
    json_file_model = _init_save_folder(save_dir)
    # 载入开始id
    coco_id = coco_start_id

    files = os.listdir(save_dir)
    for file in files:
        file_path = os.path.join(save_dir, file)
        for keystring in ["train", "val", "test"]:
            if file.find(keystring) > 0:
                mode = keystring + '2017'
                json_file = copy.deepcopy(json_file_model)

                filenames = os.listdir(file_path)
                for filename in tqdm(filenames):
                    if filename.endswith(".jpg"):
                        img_path = os.path.join(file_path, filename)
                        json_path = os.path.join(file_path, filename[:filename.rfind(".")]+".json")

                        # 加载非coco格式的json文件
                        hand_pts = json.load(open(json_path))
                        if type(hand_pts['hand_pts']) == list:
                            img = cv2.imread(img_path)

                            kp = np.array(hand_pts['hand_pts'])
                            # img, img_path, landmarks, json_file, mode, save_dir, img_id

                            flag = convert_coco_format(img, kp, json_file, mode, save_dir, coco_id)
                            if flag == 1:
                                coco_id +=1
                                # print(coco_id)

                with open(os.path.join(save_dir, 'annotations', f'person_keypoints_{mode}.json'), 'w') as fw:
                    json.dump(json_file, fw)

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
    parser.add_argument("--SavePath",
                        default=r"F:\image\CMU\hand_labels_synth\hand_labels_synth",
                        help="this parameter is about the PATH of ImgSave", dest="SavePath", type=str)
    # 解析参数
    args = parser.parse_args()
    print("parameter 'save_path' is :", args.SavePath)
    main(args.SavePath, coco_start_id=1000_000)