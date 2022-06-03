"""
重新构建v2_6的数据集，一张图片对应一个images的一个unit，每个手对应annotations的一个unit
images['id']标识图片id
annotations['id']标识手对象id
annotation['image_id']标识图片id

同时把所有关键点坐标保留两位保存

对于不同数据集的convert_coco_format仅需要对load_data做修改
对于load_data: 参照coco的读取数据方式，以原始图片名做索引号，对每张图片存储关键信息单元如下
unit_dict = dict({
                'hand_type': hand_type,
                'image_path': img_path,
                'keypoints': keypoints,
            })

YouTu 3D数据集原始数据全部都给了标注团队标注，所以对于该数据集主要输入数据是标注团队反馈回来的json文件

"""
import argparse
import os
from tqdm import tqdm
import cv2
import numpy as np
import json
import copy
from collections import defaultdict
import pickle
import imagesize

from json_tools import _init_save_folder
from dataset_v2_6 import convert_coco_format_from_wholebody
from tools import draw_2d_points

COCOBBOX_FACTOR = 1.5
COCO_START_ID = 1_300_000
VAL_NUM = 1500
Divide_1 = COCO_START_ID + VAL_NUM
Debug = 0


def main(data_dir, save_dir):
    # init
    json_file_model = _init_save_folder(save_dir)
    val_json, train_json = copy.deepcopy(json_file_model), copy.deepcopy(json_file_model)
    coco_id, image_id = COCO_START_ID, COCO_START_ID
    image_counter = 0
    hand_type = None

    # Load information using coco method
    print("loading the data......")
    data_dict, tag_names = load_data(data_dir)

    img_num = len(tag_names)
    shuffle_list = np.arange(img_num)
    rng = np.random.default_rng(12345)
    rng.shuffle(shuffle_list)

    print("processing data......")
    for index in tqdm(range(img_num)):
        shuffle_id = shuffle_list[index]
        tag_name = tag_names[shuffle_id]

        unit_list = data_dict[tag_name]
        if Debug:
            debug_image = cv2.imread(unit_list[0]['image_path'])

        flag = 0
        for hand_index, unit_dict in enumerate(unit_list):
            img_path = unit_dict['image_path']
            keypoints = unit_dict['keypoints']

            if VAL_NUM > image_counter >= 0:
                mode = 'val2017'
                json_file = val_json
            elif image_counter >= VAL_NUM:
                if image_counter == VAL_NUM and VAL_NUM != 0:
                    with open(os.path.join(save_dir, 'annotations', f'person_keypoints_val2017.json'), 'w') as fw:
                        json.dump(val_json, fw)
                mode = 'train2017'
                json_file = train_json

            flag = convert_coco_format_from_wholebody(img_path, keypoints, json_file, mode, save_dir, coco_id,
                                                      image_id, save_flag=flag)
            if Debug:
                debug_image = draw_2d_points(keypoints, debug_image)
                if hand_type is not None:
                    txt = 'l' if hand_type == 'left' else 'r'
                    debug_image = cv2.putText(debug_image, txt, (int(keypoints[0, 0]), int(keypoints[0, 1])),
                                        cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 2)

            if flag == 1:
                coco_id +=1

        if Debug:
            cv2.imshow('show', debug_image)
            cv2.waitKey(0)

        image_id += 1
        if flag == 1:
            image_counter += 1

    print(f"Writing the json file person_keypoints_train.json......")
    with open(os.path.join(save_dir, 'annotations', f'person_keypoints_train2017.json'), 'w') as fw:
        json.dump(train_json, fw)


def load_data(data_dir):
    data_dict = defaultdict(list)

    file_names = os.listdir(data_dir)
    for file_name in file_names:        # [train]
        if not os.path.isdir(os.path.join(data_dir, file_name)) or file_name not in ["train"]:
            continue

        file_path = os.path.join(data_dir, file_name)
        data_names = os.listdir(file_path)
        for data_name in tqdm(data_names):        # [ABF10, ABF11]
            data_path = os.path.join(file_path, data_name)      # ./evaluation/ABP10
            data_rgb_path = os.path.join(data_path, "rgb")

            img_names = os.listdir(data_rgb_path)
            for img_name in img_names:
                if not img_name.endswith('.jpg'):
                    continue

                # extract image path
                img_path = os.path.join(data_rgb_path, img_name)

                # extract keypoints
                mark_index = data_path[-1]
                inter_param_path = os.path.join(data_dir, "calibration", data_name[:-1],
                                                "calibration", 'cam_' + mark_index + '_intrinsics.txt')
                pickle_path = os.path.join(data_path, "meta", img_name.split(".")[0] + ".pkl")
                K = get_intrinsics(inter_param_path).tolist()
                optPickData = load_pickle_data(pickle_path)
                handJoints3D = optPickData['handJoints3D']
                if np.all(handJoints3D == 0) or np.all(handJoints3D == None):
                    continue
                kp = projectPoints(handJoints3D, K)
                keypoints = flip(canonical_coordinates(kp), img_path)

                unit_dict = dict({
                    'image_path': img_path,
                    'keypoints': keypoints,
                })

                tag_name = f'{data_name}_{img_name}'
                data_dict[tag_name].append(unit_dict)

    return data_dict, list(data_dict.keys())


def get_intrinsics(filename):
    with open(filename, 'r') as f:
        line = f.readline()
    line = line.strip()
    items = line.split(',')
    for item in items:
        if 'fx' in item:
            fx = float(item.split(':')[1].strip())
        elif 'fy' in item:
            fy = float(item.split(':')[1].strip())
        elif 'ppx' in item:
            ppx = float(item.split(':')[1].strip())
        elif 'ppy' in item:
            ppy = float(item.split(':')[1].strip())

    camMat = np.array([[fx, 0, ppx], [0, fy, ppy], [0, 0, 1]])
    return camMat


def flip(keypoints, image_path):
    width, height = imagesize.get(image_path)
    keypoints[:, 0] = width - keypoints[:, 0]
    return keypoints

def load_pickle_data(f_name):
    """ Loads the pickle data """
    if not os.path.exists(f_name):
        raise Exception('Unable to find annotations picle file at %s. Aborting.'%(f_name))
    with open(f_name, 'rb') as f:
        try:
            pickle_data = pickle.load(f, encoding='latin1')
        except:
            pickle_data = pickle.load(f)

    return pickle_data


def projectPoints(xyz, K):
    """ Project 3D coordinates into image space. """
    xyz = np.array(xyz)
    K = np.array(K)
    uv = np.matmul(K, xyz.T).T
    return uv[:] / uv[:, -1:]


def canonical_coordinates(points):
    kp = np.ones(63).reshape(21,3)

    kp[0] = points[0]
    kp[1] = points[13]
    kp[2] = points[14]
    kp[3] = points[15]
    kp[4] = points[16]
    kp[5] = points[1]
    kp[6] = points[2]
    kp[7] = points[3]
    kp[8] = points[17]
    kp[9] = points[4]
    kp[10] = points[5]
    kp[11] = points[6]
    kp[12] = points[18]
    kp[13] = points[10]
    kp[14] = points[11]
    kp[15] = points[12]
    kp[16] = points[19]
    kp[17] = points[7]
    kp[18] = points[8]
    kp[19] = points[9]
    kp[20] = points[20]
    return kp


if __name__ == "__main__":
    # 创建一个解析器
    parser = argparse.ArgumentParser()
    # 添加参数ImgDatPath
    parser.add_argument("--SaveDir", default=r"G:\imgdate2\HO3D_v3\HO3D_from_whole_body_v2_6")
    parser.add_argument("--DataDir", default=r"G:\imgdate2\HO3D_v3\HO3D_v3")
    # 解析参数
    args = parser.parse_args()
    main(args.DataDir, args.SaveDir)