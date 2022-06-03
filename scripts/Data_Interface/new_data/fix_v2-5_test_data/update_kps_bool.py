from tqdm import tqdm
import os
import numpy as np

from json_tools import load_json_data, get_ids, load_txt_data


def main():
    crop_json_path = \
        r'G:\test_data\new_data\new_data_from_whole_body\annotations_need_to_match\v2_4_person_keypoints_test2017.json'
    whole_json_path = \
        r'G:\test_data\new_data\new_data_from_whole_body\annotations\test-update.json'
    match_txt_path = r'G:\test_data\new_data\new_data_from_whole_body\match-v2_4-v2_5\match.txt'

    whole_images_dict, whole_annotations_dict = load_json_data(whole_json_path)
    crop_images_dict, crop_annotations_dict = load_json_data(crop_json_path)

    crop_image_names, whole_image_names = get_image_path(match_txt_path)
    assert (len(crop_image_names)==len(whole_image_names))

    iter_num = len(crop_image_names)
    for i in tqdm(range(iter_num)):
        crop_image_id = crop_image_names[i].split('.')[0]
        whole_image_id = whole_image_names[i].split('.')[0]

        crop_annotation_info = crop_annotations_dict[crop_image_id]
        whole_annotations_list = whole_annotations_dict[whole_image_id]

        crop_keypoints = np.array(crop_annotation_info["keypoints"]).reshape(21, 3)
        for whole_annotation_info in whole_annotations_list:
            whole_keypoints = np.array(whole_annotation_info["keypoints"]).reshape(21, 3)
            crop_keypoints_crop_box_wo_img(whole_keypoints)





def get_image_path(txt_path):
    crop_image_names, whole_image_names = [], []
    data_info_list = load_txt_data(txt_path)
    for data_info in data_info_list:
        crop_image_path, whole_image_path = data_info.split('**')
        crop_image_names.append(os.basename(crop_image_path))
        whole_image_names.append(os.basename(whole_image_path))

    return crop_image_names, whole_image_names


def crop_box_wo_img(hand_pts_2d, box_factor=2.2):
    coco_kps = hand_pts_2d.copy()

    kps_valid_bool = coco_kps[:, -1].astype(bool)
    new_hand_pts_2d = coco_kps[:, :2][kps_valid_bool]

    hand_min = np.min(new_hand_pts_2d, axis=0)
    hand_max = np.max(new_hand_pts_2d, axis=0)
    hand_box_c = (hand_max + hand_min) / 2.
    half_size = int(np.max(hand_max - hand_min) * box_factor / 2.)

    x_left = int(hand_box_c[0] - half_size)
    y_top = int(hand_box_c[1] - half_size)

    save_pts = hand_pts_2d[:, :2] - np.array([x_left, y_top])

    return save_pts


if __name__ == "__main__":
    main()
