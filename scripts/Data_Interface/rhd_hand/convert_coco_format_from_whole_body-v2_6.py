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

from library.json_tools import _init_save_folder
from library.dataset_v2_6 import convert_coco_format_from_wholebody
from library.tools import draw_2d_points

COCOBBOX_FACTOR = 1.5
COCO_START_ID = 1_100_000
VAL_NUM = 0
Divide_1 = COCO_START_ID + VAL_NUM
Debug = 0


def main(data_dir, save_dir):
    # init
    json_file_model = _init_save_folder(save_dir)
    val_json, train_json = copy.deepcopy(json_file_model), copy.deepcopy(json_file_model)
    coco_id, image_id = COCO_START_ID, COCO_START_ID
    image_counter = 1

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
            h, w, _ = debug_image.shape

        flag = 0
        for hand_index, unit_dict in enumerate(unit_list):
            img_path = unit_dict['image_path']
            keypoints = unit_dict['keypoints']
            hand_type = unit_dict['hand_type']

            if VAL_NUM > image_counter >= 0:
                mode = 'val2017'
                json_file = val_json
            elif image_counter >= VAL_NUM:
                if image_counter == VAL_NUM:
                    with open(os.path.join(save_dir, 'annotations', f'person_keypoints_val2017.json'), 'w') as fw:
                        json.dump(val_json, fw)
                mode = 'train2017'
                json_file = train_json

            flag = convert_coco_format_from_wholebody(img_path, keypoints, json_file, mode, save_dir, coco_id,
                                                      image_id, hand_type=hand_type, save_flag=flag)
            if Debug:
                debug_image = draw_2d_points(keypoints, debug_image)
                txt = 'r' if hand_type == 'right' else 'l'
                debug_image = cv2.putText(debug_image, txt, (int(keypoints[0, 0]), int(keypoints[0, 1])), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 125, 255), 2)

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

    filenames = os.listdir(data_dir)
    for filename in filenames:
        if filename not in ["evaluation", "training"]:
            continue
        pickle_dir = os.path.join(data_dir, filename, 'anno_%s.pickle' % filename)

        with open(pickle_dir, 'rb') as fi:
            anno_all = pickle.load(fi)

            # iterate samples of the set
            for sample_id, anno in tqdm(anno_all.items()):
                image_path = os.path.join(data_dir, filename, 'color', '%.5d.png' % sample_id)
                tage_name = f'{filename}_{sample_id}'

                hands = dict()
                if np.any(anno['uv_vis'][:21, 2]) == 1:
                    kp = coordinate_normalization(anno['uv_vis'][:21])
                    hands['left'] = kp
                if np.any(anno['uv_vis'][21:42, 2]) == 1:
                    kp = coordinate_normalization(anno['uv_vis'][21:42])
                    hands['right'] = kp

                for key_word in hands.keys():
                    keypoints = hands[key_word]
                    unit_dict = dict({
                        'image_path': image_path,
                        'keypoints': keypoints,
                        'hand_type': key_word
                    })
                    data_dict[tage_name].append(unit_dict)

    print("\n")
    return data_dict, list(data_dict.keys())


def coordinate_normalization(points):
    kp = np.zeros(63).reshape(21,3)
    kp[0] = points[0]
    kp[1] = points[4]
    kp[2] = points[3]
    kp[3] = points[2]
    kp[4] = points[1]
    kp[5] = points[8]
    kp[6] = points[7]
    kp[7] = points[6]
    kp[8] = points[5]
    kp[9] = points[12]
    kp[10] = points[11]
    kp[11] = points[10]
    kp[12] = points[9]
    kp[13] = points[16]
    kp[14] = points[15]
    kp[15] = points[14]
    kp[16] = points[13]
    kp[17] = points[20]
    kp[18] = points[19]
    kp[19] = points[18]
    kp[20] = points[17]
    return kp


if __name__ == "__main__":
    # 创建一个解析器
    parser = argparse.ArgumentParser()
    # 添加参数ImgDatPath
    parser.add_argument("--DataDir",
                        default=r"F:\image\Rendered Handpose Dataset Dataset\RHD\RHD_published_v2")
    parser.add_argument("--SaveDir",
                        default=r"F:\image\Rendered Handpose Dataset Dataset\RHD\RHD_from_whole_body_v2_6")
    # 解析参数
    args = parser.parse_args()
    main(args.DataDir, args.SaveDir)
