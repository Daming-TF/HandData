from .tools import get_file_list, get_keypoints
import os
import json
from tqdm import tqdm
import numpy as np

from json_tools import get_ids

data_path = r'F:\image\CMU\hand_labels\hand_labels'

def update_from_batch_json(images_dict, annotations_dict, batch_sample_path, save_dir, debug=False):
    ids = get_ids(annotations_dict)

    # 按时间序列返回json文件
    json_files = get_file_list(batch_sample_path)
    for json_file in json_files:
        batch_sample_dir = os.path.join(batch_sample_path, json_file)
        with open(batch_sample_dir, 'r', encoding='UTF-8')as f:
            batch_sample_data = json.load(f)
        print(f'There is >{ len(batch_sample_data) }< tag pic need to update')

        # 遍历打回数据的信息，匹配原图路径
        for index in tqdm(range(len(batch_sample_data))):
            tag_dict = batch_sample_data[index]
            label_feature = tag_dict['labelFeature']
            original_filename = tag_dict['originalFileName']

            # 根据打回数据，得到对应原图路径
            image_info = original_filename.split('_')
            file_name = 'manual_' + image_info[2]
            image_name = original_filename[original_filename.find(f'{file_name}') + len(file_name) + 1:]
            image_dir = os.path.join(data_path, file_name, image_name)

            same_img_flag = 0

            for i, image_id in enumerate(ids):
                images_info = images_dict[image_id]
                original_dir = images_info['image_dir']
                if original_dir == image_dir:
                    same_img_flag = 1
                    break

            if not same_img_flag:
                # print(f"There is not exist a same pic with {image_dir}")
                continue

            handlandmarks_list = get_keypoints(label_feature)
            new_annotation_unit = creat_annotation_unit(handlandmarks_list)

    if debug:
        with open(save_dir, 'w') as fw:
            json.dump(json_data, fw)
            print("train2017.json have succeed to write")


def creat_annotation_unit(handlandmarks_list):
    for handlandmarks in handlandmarks_list:


def get_annotation_info(coco_kps):
    kps_valid_bool = coco_kps[:, -1].astype(np.bool)
    key_pts = coco_kps[:, :2][kps_valid_bool]

    hand_min = np.min(key_pts, axis=0)  # (2,)
    hand_max = np.max(key_pts, axis=0)  # (2,)
    hand_box_c = (hand_max + hand_min) / 2  # (2, )
    half_size = int(np.max(hand_max - hand_min) * 1.5 / 2.)  # int

    x_left = int(hand_box_c[0] - half_size)
    y_top = int(hand_box_c[1] - half_size)
    x_right = x_left + 2 * half_size
    y_bottom = y_top + 2 * half_size
    box_w = x_right - x_left
    box_h = y_bottom - y_top
    if min(box_h, box_w) < 48:
        return 0

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
        'hand_type': hand_type
    })
