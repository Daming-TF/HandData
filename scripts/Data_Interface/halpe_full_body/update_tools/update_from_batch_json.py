"""
这里是针对标注团队反馈回来的批注数据进行重覆盖， 但要注意对于CMU-real每个手是一张图片
所以在加载数据的时候也要用coco形式加载，以原始图片路径作为索引号，不能直接覆盖，不然导致双手图片在覆盖后只剩下一个

另外功能：
1. 在配对图片成功后会在保存json文件路径下创建一个{mode}_batch_data文件夹存放配对成功
    mode ∈ ['train', 'val', 'test']
    每次运行程序时都会先清空记录文件再记录
2. 当不需要保留这个batch simple覆盖的json文件，把输入参数debug设为Fasle即可
"""

import os
import json
from tqdm import tqdm
import numpy as np
from collections import defaultdict

from library.json_tools import get_ids, write_json
from .convert_tools import get_file_list, get_keypoints

data_path = r'E:\Data\landmarks\HFB\halpe_Full-Body_Human_Keypoints_and_HOL-Det_dataset'


def update_from_batch_json(images_dict, annotations_dict, batch_sample_path, save_path, debug=False):
    # counter the number of times txt is opened
    counter = 0
    # get image ids
    ids = get_ids(annotations_dict)

    # Sort json files according to time
    json_files = get_file_list(batch_sample_path)
    for json_file in json_files:
        batch_sample_dir = os.path.join(batch_sample_path, json_file)
        with open(batch_sample_dir, 'r', encoding='UTF-8')as f:
            batch_sample_data = json.load(f)

        print(f'There is >{ len(batch_sample_data) }< tag pic need to update')

        # Traverse the information of the returned data and match the original image path
        for index in tqdm(range(len(batch_sample_data))):
            tag_dict = batch_sample_data[index]
            label_feature = tag_dict['labelFeature']
            original_filename = tag_dict['originalFileName']

            # extract the image path
            image_info = original_filename.split('_')
            file_name = image_info[1]
            if file_name == 'val2017':
                image_name = image_info[2]
                image_dir = os.path.join(data_path, file_name, image_name)
            else:
                image_name = image_info[4]
                image_dir = os.path.join(data_path, 'hico_20160224_det', 'images', 'train2015',
                                         'HICO_train2015_'+image_name)

            match_flag = 0

            for i, image_id in enumerate(ids):
                images_info = images_dict[image_id]
                original_dir = images_info['image_dir']
                if original_dir == image_dir:
                    match_flag = 1
                    break

            if not match_flag:
                # print()
                continue
            record_match(image_id, index, save_path, counter)
            counter += 1
            handlandmarks_list = get_keypoints(label_feature)
            annotations_dict = update(annotations_dict, handlandmarks_list, image_id, get_start_coco_id(annotations_dict))

    if debug:
        write_json(images_dict, annotations_dict, save_path)
        print(f"There are >>{counter}<< data match")


# def load_anno_team_data(anno_team_json_data):
#     annotations_dict = defaultdict(list)
#     images_dict = dict()
#
#     for anno_team_json_unit in anno_team_json_data:
#         original_filename = anno_team_json_unit['originalFileName']
#         # extract the image path
#         image_info = original_filename.split('_')
#         file_name = 'manual_' + image_info[2]
#         image_name = original_filename[original_filename.find(f'{file_name}') + len(file_name) + 1:]
#         image_dir = os.path.join(data_path, file_name, image_name)
#
#         for split_name in ['l', 'r']:
#             if image_name.find(split_name + '.jpg') < 0:
#                 continue
#             tag_name = image_name.split(split_name + '.jpg')[0]


def update(annotations_dict, handlandmarks_list, image_id, start_coco_id):
    coco_id = start_coco_id
    annotations_info_list = []
    for hand_landmarks in handlandmarks_list:
        kps_valid_bool = hand_landmarks[:, -1].astype(bool)
        key_pts = hand_landmarks[:, :2][kps_valid_bool]

        hand_min = np.min(key_pts, axis=0)  # (2,)
        hand_max = np.max(key_pts, axis=0)  # (2,)
        hand_box_c = (hand_max + hand_min) / 2
        half_size = int(np.max(hand_max - hand_min) * 1.5 / 2.)  # int

        x_left = int(hand_box_c[0] - half_size)
        y_top = int(hand_box_c[1] - half_size)
        x_right = x_left + 2 * half_size
        y_bottom = y_top + 2 * half_size
        box_w = x_right - x_left
        box_h = y_bottom - y_top
        if min(box_h, box_w) < 48:
            continue

        coco_kps = hand_landmarks.flatten().tolist()
        anno_dict = dict({
            'segmentation': [[x_left, y_top, x_right, y_top, x_right, y_bottom, x_left, y_bottom]],
            'num_keypoints': 21,
            'area': box_h * box_w,
            'iscrowd': 0,
            'keypoints': coco_kps,
            'image_id': image_id,
            'bbox': [x_left, y_top, box_w, box_h],  # 1.5 expand
            'category_id': 1,
            'id': coco_id,
            'hand_type': None
        })
        annotations_info_list.append(anno_dict)
        coco_id += 1

    if len(annotations_info_list) != 0:
        annotations_dict[image_id] = annotations_info_list

    return annotations_dict


def get_start_coco_id(annotations_dict):
    ids = get_ids(annotations_dict)
    final_image_id = ids[-1]
    return (annotations_dict[final_image_id][-1]['id'])+1


def record_match(image_id, index, save_path, counter):
    mode = None
    for keyword in ['train', 'val']:
        if keyword in save_path:
            mode = keyword
            break

    save_dir = os.path.join(os.path.split(save_path)[0], f'{mode}_batch_data')
    save_path = os.path.join(save_dir, 'record.txt')
    os.makedirs(save_dir, exist_ok=True)
    if counter == 0:
        open(save_path, 'w').close()
    with open(save_path, 'a')as f:
        f.write(f"{image_id}**{index}\n")
