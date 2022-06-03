"""
由于v2_5每张图片存在像素值差距难以和v2_4的数据集匹配上
所有通过该程序把v2_5左右手信息加入到v2_4
"""

import json
from collections import defaultdict
import numpy as np
from tqdm import tqdm

from json_tools import make_json_head, crop_box_wo_img

whole_json_path = r'G:\test_data\new_data\new_data_from_whole_body\annotations\test_w_hand_type.json'
crop_json_path = r'G:\test_data\new_data\new_data_from_whole_body\match-v2_4-v2_5\v2_4_person_keypoints_test2017.json'
save_path = r'G:\test_data\new_data\new_data_from_whole_body\match-v2_4-v2_5\v2_4_person_keypoints_test2017-update.json'

def main():
    whole_images_dict, whole_annotations_dict = load_json_data(whole_json_path)
    crop_images_dict, crop_annotations_dict = load_json_data(crop_json_path)

    whole_ids = get_ids(whole_images_dict)
    crop_ids = get_ids(crop_images_dict)

    for crop_id in tqdm(crop_ids):
        crop_annotation_info = crop_annotations_dict[crop_id][0]
        crop_keypoints = np.array(crop_annotation_info["keypoints"]).reshape(21, 3)

        flag = 0

        for whole_id in whole_ids:
            whole_annotation_info_list = whole_annotations_dict[whole_id]

            for whole_annotation_info in whole_annotation_info_list:
                whole_keypoints = np.array(whole_annotation_info["keypoints"]).reshape(21, 3)

                crop_keypoints_from_whole = np.zeros([21, 3])
                crop_keypoints_from_whole[:, 2] = whole_keypoints[:, 2]
                crop_keypoints_from_whole[:, :2] = crop_box_wo_img(whole_keypoints)

                if get_match(crop_keypoints, crop_keypoints_from_whole):
                    crop_annotation_info["hand_type"] = whole_annotation_info["hand_type"]
                    flag = 1

            if flag:
                break

    check_no_hand(crop_annotations_dict)
    write_json(crop_images_dict, crop_annotations_dict, save_path)


def check_no_hand(crop_annotations_dict):
    crop_ids = get_ids(crop_annotations_dict)
    for crop_id in crop_ids:
        crop_annotation_info = crop_annotations_dict[crop_id][0]
        if "hand_type" not in crop_annotation_info.keys():
            crop_annotation_info["hand_type"] = "right"


def get_match(kps1, kps2):
    kps_flag = False
    kps1 = kps1[:, :2].flatten()
    kps2 = kps2[:, :2].flatten()

    if np.sum(np.abs(kps1-kps2)) == 0:
        kps_flag = True

    return kps_flag


def write_json(images_dict, annotations_dict, save_path):
    json_head = make_json_head()
    images_ids = get_ids(images_dict)
    for image_id in images_ids:
        image_info = images_dict[image_id]
        json_head["images"].append(image_info)

        annotation_info_list = annotations_dict[image_id]
        for annotation_info in annotation_info_list:
            json_head["annotations"].append(annotation_info)

    with open(save_path, 'w')as f:
        json.dump(json_head, f)


def convert_box(bbox):
    x_left, y_top, box_w, box_h = bbox
    return [x_left, y_top, x_left+box_w, y_top+box_h]


def load_json_data(json_path):
    annotations_dict = defaultdict(list)
    images_dict = {}
    with open(json_path, 'r')as f:
        dataset = json.load(f)

    for ann in dataset['annotations']:
        annotations_dict[ann['image_id']].append(ann)

    for img in dataset['images']:
        images_dict[img['id']] = img

    return images_dict, annotations_dict


def get_ids(data_dict):
    return list(data_dict.keys())


if __name__ == '__main__':
    main()