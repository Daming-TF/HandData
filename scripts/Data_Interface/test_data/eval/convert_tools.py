import numpy as np
import cv2
import os

from library.tools import draw_2d_points

NUM_JOINTS = 21

COCOBBOX_FACTOR = 1.5
CROP_FACTOR = 2.2
MIN_SIZE = 48
DATA_CAPTURED = '2022-03-01 11:11:11'
NUM_JOINTS = 21


def refine_keypts(key_pts):
    coco_kps = np.zeros((21, 3), dtype=key_pts.dtype)
    coco_kps[:, :2] = key_pts
    coco_kps[:, 2] = 2
    for idx in range(coco_kps.shape[0]):
        if coco_kps[idx, 0] <= -50 and coco_kps[idx, 1] <= -50:
            coco_kps[idx] = 0
        if np.all(coco_kps[idx, :2] == 0):
            coco_kps[idx, 2] = 0
    return coco_kps


def convert_coco_format_from_wholebody(json_total, json_file, img, coco_id, image_id, landmarks,
        coco_factor=COCOBBOX_FACTOR):     # json_head,image_dir,image_id,file_name,keypoint
    file_name = str(image_id).zfill(12)+'.jpg'
    img_h, img_w = img.shape[:2]
    key_pts = landmarks[:, :2].copy()  # exclude z-coordinate
    assert isinstance(coco_id, int)

    # *** add 8.26 ***
    coco_kps = refine_keypts(key_pts)
    kps_valid_bool = coco_kps[:, -1].astype(np.bool)
    key_pts = coco_kps[:, :2][kps_valid_bool]
    # *** add 8.26 ***

    hand_min = np.min(key_pts, axis=0)      # (2,)
    hand_max = np.max(key_pts, axis=0)      # (2,)
    hand_box_c = (hand_max + hand_min) / 2  # (2, )
    half_size = int(np.max(hand_max - hand_min) * coco_factor / 2.)  # int

    x_left = int(hand_box_c[0] - half_size)
    y_top = int(hand_box_c[1] - half_size)
    x_right = x_left + 2 * half_size
    y_bottom = y_top + 2 * half_size
    box_w = x_right - x_left
    box_h = y_bottom - y_top
    if min(box_h, box_w) < 48:
        print(f'TOO SMALL! img_path: {file_name}')
        print(f'box_h: {box_h}, box_w: {box_w}\n')
        return 0

    image_dict = dict({
        'license': 1,
        'file_name': file_name,
        'coco_url': 'Unavailable',
        'height': img_h,
        'width': img_w,
        'date_captured': '2022-03-01 11:11:11',
        'flickr_url': 'Unavailable',
        'id': image_id,
    })

    # points, im_ori

    coco_kps = coco_kps.flatten().tolist()
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
        'score': 1
    })

    json_file['images'].append(image_dict)
    json_file['annotations'].append(anno_dict)
    json_total['images'].append(image_dict)
    json_total['annotations'].append(anno_dict)

    # crop_img = draw_point(key_pts, crop_img)   # show landmarks to check the landmarks correct or not
    return 1
