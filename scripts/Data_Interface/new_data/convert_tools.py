import numpy as np
import cv2
import os
from library.tools import draw_2d_points

NUM_JOINTS = 21

COCOBBOX_FACTOR = 1.5
CROP_FACTOR = 2.2
MIN_SIZE = 48
DATA_CAPTURED = '2022-04-04 11:11:11'
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


def convert_coco_format_from_wholebody(img_path, landmarks, json_file, hand_id, img_id, mode=None, save_dir=None, coco_factor=COCOBBOX_FACTOR):
    # frame_name = img_path.split('/')[-1]
    # folder_name = img_path.split('/')[-4]
    file_name = str(img_id).zfill(12)+'.jpg'

    img = cv2.imread(img_path)
    img_h, img_w = img.shape[:2]
    key_pts = landmarks[:, :2].copy()  # exclude z-coordinate
    assert isinstance(img_id, int)

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
    if min(box_h, box_w) < MIN_SIZE:
        print(f'TOO SMALL! img_path: {img_path}')
        print(f'box_h: {box_h}, box_w: {box_w}\n')
        return 0

    if mode != None and save_dir != None:
        save_path = os.path.join(save_dir, 'images', f'{mode}')
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        save_path = os.path.join(save_path, file_name)
        cv2.imshow('a', draw_2d_points(landmarks, img))
        cv2.waitKey(1)
        cv2.imwrite(save_path, img)

    image_dict = dict({
        'license': 1,
        'file_name': file_name,
        'image_dir': img_path,
        'coco_url': 'Unavailable',
        'height': img_h,
        'width': img_w,
        'date_captured': DATA_CAPTURED,
        'flickr_url': 'Unavailable',
        'id': hand_id
    })

    coco_kps = coco_kps.flatten().tolist()
    anno_dict = dict({
        'segmentation': [[x_left, y_top, x_right, y_top, x_right, y_bottom, x_left, y_bottom]],
        'num_keypoints': NUM_JOINTS,
        'area': box_h * box_w,
        'iscrowd': 0,
        'keypoints': coco_kps,
        'image_id': img_id,
        'bbox': [x_left, y_top, box_w, box_h],  # 1.5 expand
        'category_id': 1,
        'id': img_id
    })

    json_file['images'].append(image_dict)
    json_file['annotations'].append(anno_dict)

    # crop_img = draw_point(key_pts, crop_img)   # show landmarks to check the landmarks correct or not
    return 1
