import cv2
import os
import numpy as np

from library.json_tools import crop_box
from library.tools import  draw_2d_points

COCOBBOX_FACTOR = 1.5
CROP_FACTOR = 2.2
MIN_SIZE = 48
DATA_CAPTURED = '2021-9-26 14:52:28'
NUM_HAND_KEYPOINTS = 21


def convert_coco_format(img_dir, landmarks, json_file, mode, save_path, img_id, img_name = '0'):
    '''

    Parameters
    ----------
    img : original
    img_path : the path of original
    landmarks : a 21*3 matrix
    json_file : a Dictionary used to store information
    mode : Three classification modes in "test", "train" and "val"
    save_dir :  the image save path
    img_id  : the image renamed id

    Returns
    -------

    '''
    img = cv2.imread(img_dir)
    # 规范图片名
    file_name = str(img_id).zfill(12) + '.jpg'

    # 实际只用上了img, landmarks, box_factor
    # print(landmarks.shape)
    # 返回的是剪辑好的照片，以及基于剪辑后坐标系的landmarks
    crop_img, crop_landmarks = crop_box(img, landmarks.copy(), box_factor=CROP_FACTOR)
    img_h, img_w = crop_img.shape[:2]
    assert isinstance(img_id, int)

    # *** add 8.26 ***
    # 重塑landmarks，把第1,2列换为以crop为边界的crop_landmarks,把第三列数值全转化为0
    coco_kps = landmarks.copy()
    coco_kps[:, :2] = crop_landmarks[:, :2]
    coco_kps[:, 2] = landmarks[:, 2]
    kps_valid_bool = coco_kps[:, -1].astype(bool)
    coco_kps[~kps_valid_bool, :2] = 0
    key_pts = coco_kps[:, :2][kps_valid_bool]
    # *** add 8.26 ***

    coco_factor = COCOBBOX_FACTOR
    hand_min = np.min(key_pts, axis=0)  # (2,)
    hand_max = np.max(key_pts, axis=0)  # (2,)
    hand_box_c = (hand_max + hand_min) / 2  # (2, )
    half_size = int(np.max(hand_max - hand_min) * coco_factor / 2.)  # int

    x_left = int(hand_box_c[0] - half_size)
    y_top = int(hand_box_c[1] - half_size)
    x_right = x_left + 2 * half_size
    y_bottom = y_top + 2 * half_size
    box_w = x_right - x_left
    box_h = y_bottom - y_top
    if min(box_h, box_w) < MIN_SIZE:
        # print(f'TOO SMALL! img_path: {img_path}')
        # print(f'box_h: {box_h}, box_w: {box_w}\n')
        return 0

    image_dict = dict({
        'license': 1,
        'original_name': img_name,
        'file_name': file_name,
        'coco_url': 'Unavailable',
        'height': img_h,
        'width': img_w,
        'date_captured': DATA_CAPTURED,
        'flickr_url': 'Unavailable',
        'original_dir': img_dir,
        'id': img_id
    })

    # img_test = draw_2d_points(coco_kps, crop_img)
    # cv2.imshow('test', img_test)
    # cv2.waitKey(0)

    coco_kps = coco_kps.flatten().tolist()
    # segmentation分别表示左上，右上，右下，左下四个点坐标
    anno_dict = dict({
        'segmentation': [[x_left, y_top, x_right, y_top, x_right, y_bottom, x_left, y_bottom]],
        'num_keypoints': NUM_HAND_KEYPOINTS,
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


    save_dir = os.path.join(save_path, 'images', f'{mode}', file_name)
    cv2.imwrite(save_dir, crop_img)

    return 1
