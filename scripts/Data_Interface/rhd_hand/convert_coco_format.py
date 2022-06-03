import argparse
import os
import numpy as np
from json_tools import _init_save_folder, convert_coco_format
import json
import cv2
import copy
from tqdm import tqdm
import pickle
from vis_hand import coordinate_normalization
from tools import mkdir

COCOBBOX_FACTOR = 1.5
COCO_START_ID = 1_100_000
TRAIN_NUM, VAL_NUM, TEST_NUM = 40_986, 1_500, 1_500

Divide_1 = COCO_START_ID + TEST_NUM
Divide_2 = Divide_1 + VAL_NUM
# END = Divide_2 + TEST_NUM

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
        if not os.path.isdir(os.path.join(data_dir, filename)) or filename not in ["evaluation", "training"]:
            continue
        pickle_dir = os.path.join(data_dir, filename, 'anno_%s.pickle' %filename)

        with open(pickle_dir, 'rb') as fi:
            anno_all = pickle.load(fi)

            # iterate samples of the set
            for sample_id, anno in tqdm(anno_all.items()):
                img_path = os.path.join(data_dir, filename, 'color', '%.5d.png' % sample_id)

                # # get info from annotation dictionary
                # kp_coord_uv = anno['uv_vis'][:, :2]  # u, v coordinates of 42 hand keypoints, pixel
                kp_visible = (anno['uv_vis'][:, 2] == 1)  # visibility of the keypoints, boolean
                # kp_coord_xyz = anno['xyz']  # x, y, z coordinates of the keypoints, in meters
                # camera_intrinsic_matrix = anno['K']  # matrix containing intrinsic parameters
                #
                # # Project world coordinates into the camera frame
                # kp_coord_uv_proj = np.matmul(kp_coord_xyz, np.transpose(camera_intrinsic_matrix))
                # kp_coord_uv_proj = kp_coord_uv_proj[:, :2] / kp_coord_uv_proj[:, 2:]

                #kp = coordinate_normalization(kp_coord_uv_proj[:21])
                a = anno['uv_vis']
                hands = list()
                if np.any(anno['uv_vis'][:21, 2]) ==1:
                    kp = coordinate_normalization(anno['uv_vis'][:21])
                    hands.append(kp)
                if np.any(anno['uv_vis'][21:42, 2]) ==1:
                    kp = coordinate_normalization(anno['uv_vis'][21:42])
                    hands.append(kp)

                for hand in hands:
                    kp = hand

                    if coco_id >= COCO_START_ID and coco_id < Divide_1:
                        mode = 'test2017'
                    elif coco_id >= Divide_1 and coco_id < Divide_2:
                        mode = 'val2017'
                    elif coco_id >= Divide_2:
                        mode = 'train2017'

                    if type(kp) == np.ndarray:
                        # print(sample_id)
                        a = os.path.exists(img_path)
                        if not os.path.exists(img_path) or np.all(kp[2]) == 0:
                            continue
                        img = cv2.imread(img_path)
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
                        default=r"F:\image\Rendered Handpose Dataset Dataset\RHD_v1-1\RHD_published_v2",
                        help="this parameter is about the PATH of ImgSave", dest="SavePath", type=str)
    parser.add_argument("--DataPath",
                        default=r"F:\image\Rendered Handpose Dataset Dataset\RHD_v1-1\RHD_published_v2",
                        help="this parameter is about the PATH of ImgSave", dest="DataPath", type=str)
    # 解析参数
    args = parser.parse_args()
    print("parameter 'save_path' is :", args.SavePath)
    print("parameter 'data_path' is :", args.DataPath)
    main(args.SavePath, args.DataPath, coco_start_id=COCO_START_ID)