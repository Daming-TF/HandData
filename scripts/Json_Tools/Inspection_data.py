import json
import cv2
from tqdm import tqdm
import os
from tools import draw_2d_points
import numpy as np

mode_list = ['test', 'train', 'val']
data_path_list = [
    r'E:\Data\landmarks\YouTube3D\YouTube3D-wholebody',
    r'E:\Data\landmarks\HFB\HFB_from_whole_body',
    r'E:\Data\landmarks\handpose_x_gesture_v1\HXG_from_whole_body',
    r'E:\Data\landmarks\FH\FH_from_whole_body',
    r'F:\image\CMU\hand_labels\hand_labels_from_whole_body',
    r'F:\image\CMU\hand143_panopticdb\hand143_pannopticdb_from_whole_body',
    r'F:\image\CMU\hand_labels_synth\hand_labels_synth_from_whole_body',
    r'F:\image\Rendered Handpose Dataset Dataset\RHD\RHD_from_whole_body',
    r'F:\image\COCO_whole_body\coco_from_whole_body',
    r'G:\imgdate2\HO3D_v3\HO3D_from_whole_body',
    r'G:\test_data\new_data\new_data_from_whole_body',      # 10
    r'G:\test_data\hardcase_data\hardcase_from_whole_body',
             ]

# json_dir_list = [
#     r'E:\whole_body_data\data_json\youtu3d',
#     r'E:\whole_body_data\data_json\halpe_fullbody',
#     r'E:\whole_body_data\data_json\hxg',
#     r'E:\whole_body_data\data_json\fh',
#     r'E:\whole_body_data\data_json\halpe_fullbody',
#     r'E:\whole_body_data\data_json\cmu-real'
#     r'E:\whole_body_data\data_json\cmu-synth'
#     r'E:\whole_body_data\data_json\coco'
#     r''
# ]
#
# json_name_list = [
#     fr'{mode}2017-update-for-invaliddata.json',
#     fr'person_keypoints_{mode}2017-update-for-invaliddata.json',
#     fr'person_keypoints_{mode}2017-update-for-invaliddata.json',
#     fr'person_keypoints_{mode}2017-update-for-invaliddata.json',
#     fr'person_keypoints_{mode}2017-update-for-invaliddata.json',
# ]


def get_data_name_index(image_id):
    if 300_000 <= image_id < 400_000:
        return 0
    elif 400_000 <= image_id < 500_000:
        return 1
    elif 500_000 <= image_id < 600_000:
        return 2
    elif 600_000 <= image_id < 800_000:
        return 3
    elif 800_000 <= image_id < 900_000:
        return 4
    elif 900_000 <= image_id < 1_000_000:
        return 5
    elif 1_000_000 <= image_id < 1_100_000:
        return 6
    elif 1_100_000 <= image_id < 1_200_000:
        return 7
    elif 1_200_000 <= image_id < 1_300_000:
        return 8
    elif 1_300_000 <= image_id < 1_400_000:
        return 9
    elif 1_400_000 <= image_id < 1_500_000:
        return 10
    elif 1_500_000 <= image_id < 1_600_000:
        return 11


def main():
    for mode in mode_list:
        json_dir = fr'E:\left_hand_label_data\annotations\person_keypoints_{mode}2017.json'

        with open(json_dir, 'r')as f:
            json_data = json.load(f)
            annotations = json_data['annotations']
            images = json_data['images']

        iter_num = len(annotations)

        for i in tqdm(range(iter_num)):
            image_info = images[i]
            annotation_info = annotations[i]

            file_name = image_info['file_name']

            image_id = annotation_info['image_id']
            index = get_data_name_index(image_id)
            data_path = data_path_list[index]

            image_dir = os.path.join(data_path, "images", f"{mode}2017", file_name)

            if not os.path.exists(image_dir):
                print(f"{file_name}:error")


if __name__ == '__main__':
    main()