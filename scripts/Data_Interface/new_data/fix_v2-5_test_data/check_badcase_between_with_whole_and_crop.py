"""
目前v2_4和v2_5的数据对不上(crop和whole两个数据集对不上)，该程序可以查看whole和crop的badcase是否能对上
"""
import json
import os.path
from collections import defaultdict

import numpy as np
from tqdm import tqdm
from models.json_tools import crop_box
from models.tools import draw_2d_points
import cv2


def load_txt_data(txt_dir):
    data_list = list()
    with open(txt_dir, 'r') as f_txt:
        info_list = f_txt.readlines()

    for i in range(len(info_list)):
        info = info_list[i].split('\n')[0]
        data_list.append(info)

    return data_list


def load_json_data(json_dir):
    with open(json_dir, 'r')as f:
        json_data = json.load(f)
        images = json_data['images']
        annotations = json_data['annotations']

    annotations_dict, images_dict = defaultdict(list), defaultdict(list)

    iter_num = len(images)
    for i in tqdm(range(iter_num)):
        image_info = images[i]
        annotation_info = annotations[i]
        assert (image_info['id'] == annotation_info['id'])

        # hight, width = image_info['height'], image_info['width']
        if "image_dir" in image_info.keys():
            file_name = image_info['file_name']
            image_dir = os.path.join(r"G:\test_data\new_data\new_data_from_whole_body\images\test2017", file_name)
        else:
            file_name = image_info['file_name']
            image_dir = os.path.join(r"G:\test_data\new_data\crop_images", file_name)
        image_id = annotation_info['image_id']
        keypoints = annotation_info['keypoints']
        images_dict[f'{image_id}'].append(image_dir)
        annotations_dict[f'{image_id}'].append(keypoints)

    return images_dict, annotations_dict


def check_dir(image_dir1, image_dir2):
    if image_dir1 == image_dir2[0]:
        return 1
    else:
        return 0


def main():
    cv2.namedWindow("test", cv2.WINDOW_NORMAL)
    crop_json_dir = r'G:\test_data\new_data\person_keypoints_test2017.json'
    whole_json_dir = r'G:\test_data\new_data\new_data_from_whole_body\annotations\person_keypoints_test2017.json'
    whole_badcase_txt_dir = r'G:\test_data\new_data\new_data_from_whole_body\weed_out_badcase\whold-body-badcase.txt'


    print(f"loading the '{crop_json_dir}'......")
    crop_images_dict, crop_annotations_dict = load_json_data(crop_json_dir)
    print(f"loading the '{whole_json_dir}'......")
    whole_images_dict, whole_annotations_dict = load_json_data(whole_json_dir)

    # print(f"make the match badcase dir")
    dir_match_list = load_txt_data(whole_badcase_txt_dir)

    whole_image_dirs, crop_image_dirs = [], []
    for i in range(len(dir_match_list)):
        whole_image_dir, crop_image_dir = dir_match_list[i].split('**')
        whole_image_dirs.append(whole_image_dir)
        crop_image_dirs.append(crop_image_dir)
    assert (len(whole_image_dirs) == len(crop_image_dirs))

    iter_num = len(crop_image_dirs)

    for i in range(iter_num):
        crop_image_dir = crop_image_dirs[i]
        crop_image_id = int(os.path.basename(crop_image_dir).split('.')[0])
        crop_keypoints = crop_annotations_dict[f"{crop_image_id}"]
        crop_image = cv2.imread(crop_image_dir)
        if not check_dir(crop_image_dir, crop_images_dict[f"{crop_image_id}"]):
            print("ERROR! the image dir in txt is different with the dir in json")
            break

        whole_image_dir = whole_image_dirs[i]
        whole_image_id = int(os.path.basename(whole_image_dir).split('.')[0])
        whole_keypoints = whole_annotations_dict[f"{whole_image_id}"]
        whole_image = cv2.imread(whole_image_dir)
        if not check_dir(whole_image_dir, whole_images_dict[f"{whole_image_id}"]):
            print("ERROR! the image dir in txt is different with the dir in json")
            break

        whole_to_crop_image, whole_to_crop_pts = crop_box(whole_image, np.array(whole_keypoints).reshape(21, 3))

        whole_to_crop_image = draw_2d_points(whole_to_crop_pts, whole_to_crop_image)
        crop_image = draw_2d_points(np.array(crop_keypoints).reshape(21, 3), crop_image)

        if crop_image.shape != whole_to_crop_image.shape:
            whole_to_crop_image = cv2.resize(whole_to_crop_image, crop_image.shape, cv2.INTER_LINEAR)

        canve = np.hstack([crop_image, whole_to_crop_image])
        cv2.imshow("test", canve)
        cv2.waitKey(0)


if __name__ == '__main__':
    main()