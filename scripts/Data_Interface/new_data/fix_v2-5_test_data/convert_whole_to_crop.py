import json
import numpy as np
import cv2
import os
from collections import defaultdict
from tqdm import tqdm


def crop_box(img, hand_pts_2d, box_factor=2.2):
    coco_kps = hand_pts_2d.copy()

    kps_valid_bool = coco_kps[:, -1].astype(np.bool)
    new_hand_pts_2d = coco_kps[:, :2][kps_valid_bool]

    hand_min = np.min(new_hand_pts_2d, axis=0)
    hand_max = np.max(new_hand_pts_2d, axis=0)
    hand_box_c = (hand_max + hand_min) / 2.
    half_size = int(np.max(hand_max - hand_min) * box_factor / 2.)

    x_left = int(hand_box_c[0] - half_size)
    y_top = int(hand_box_c[1] - half_size)
    x_right = x_left + 2 * half_size
    y_bottom = y_top + 2 * half_size

    pts_save = hand_pts_2d[:, :2] - np.array([x_left, y_top])

    crop_size = 2 * half_size
    img_save = np.zeros((crop_size, crop_size, 3), dtype=img.dtype)
    x_start = 0 if x_left >= 0 else -x_left
    y_start = 0 if y_top >= 0 else -y_top
    x_end = crop_size if x_right < img.shape[1] else crop_size - (x_right - img.shape[1])
    y_end = crop_size if y_bottom < img.shape[0] else crop_size - (y_bottom - img.shape[0])

    x_left = max(x_left, 0)
    y_top = max(y_top, 0)
    x_right = min(x_right, img.shape[1])
    y_bottom = min(y_bottom, img.shape[0])
    if not img_save[y_start:y_end, x_start:x_end].shape == img[y_top:y_bottom, x_left:x_right].shape:
        return img, pts_save
    img_save[y_start:y_end, x_start:x_end] = img[y_top:y_bottom, x_left:x_right]

    return img_save, pts_save


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

        image_id = annotation_info['image_id']
        images_dict[f'{image_id}'].append(image_info)
        annotations_dict[f'{image_id}'].append(annotation_info)

    return images_dict, annotations_dict


def get_image_id(data_dict):
    return list(data_dict.keys())


def main():
    crop_json_dir = r''
    whole_json_dir = r''
    print(f"loading the '{crop_json_dir}'......")
    crop_images_dict, crop_annotations_dict = load_json_data(crop_json_dir)
    print(f"loading the '{whole_json_dir}'......")
    whole_images_dict, whole_annotations_dict = load_json_data(whole_json_dir)

    image_id_list = get_image_id(crop_images_dict)
    for image_id in image_id_list:
        whole_image_info_list = whole_images_dict[f'{image_id}']
        whole_annotation_info_lsit = whole_annotations_dict[f'{image_id}']

        for whole_image_info, whole_annotation_info in zip(whole_image_info_list, whole_annotation_info_lsit):
            file_name = whole_image_info['file_name']
            image_dir = os.path.join(r"G:\test_data\new_data\new_data_from_whole_body\images\test2017", file_name)
            image = cv2.imread(image_dir)

            keypoints = whole_image_info['keypoints']
            whole_to_crop_image, whole_to_crop_pts = crop_box(image, keypoints)      # img, hand_pts_2d
            w_h, w_w = whole_to_crop_image.shape[:2]

            crop_image_info_list = crop_images_dict[f'{image_id}']
            for crop_image_info in crop_image_info_list:
                c_h, c_w = crop_image_info['height', 'width']







if __name__ == "__main__":
    main()