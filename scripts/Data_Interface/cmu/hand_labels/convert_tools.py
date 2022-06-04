import pickle
import numpy as np
import cv2
import os
import json
from tqdm import tqdm

from library.json_tools import _init_save_folder
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


def convert_coco_format_from_wholebody(img_path, landmarks, json_file, mode, save_dir, img_id, coco_factor=COCOBBOX_FACTOR):
    img = cv2.imread(img_path)
    file_name = str(img_id).zfill(12) + '.jpg'

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

    save_path = os.path.join(save_dir, 'images', f'{mode}', file_name)

    image_dict = dict({
        'license': 1,
        'file_name': file_name,
        'image_dir': img_path,
        'coco_url': 'Unavailable',
        'height': img_h,
        'width': img_w,
        'date_captured': DATA_CAPTURED,
        'flickr_url': 'Unavailable',
        'id': img_id
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
    # cv2.imshow('a', draw_2d_points(landmarks, img))
    # cv2.waitKey(0)
    cv2.imwrite(save_path, img)
    return 1


def _init_mano(mano_left_path, mano_right_path, verbose=True):
    with open(mano_left_path, 'rb') as lhand_file:
        lhand_data = pickle.load(lhand_file, encoding='latin1')

    with open(mano_right_path, 'rb') as rhand_file:
        rhand_data = pickle.load(rhand_file, encoding='latin1')

    if verbose:
        print(" [!] Check Key and Value in MANO model")
    for key, value in lhand_data.items():
        if isinstance(value, str):
            if verbose:
                print(f'{key}:, {value} - value')
        else:
            if verbose:
                print(f'{key}, {value.shape} - shape')

    lhand_reg = lhand_data['J_regressor'].toarray()  # from csc_matrix to numpy array
    rhand_reg = rhand_data['J_regressor'].toarray()  # from csc_matrix to numpy array

    return lhand_reg, rhand_reg


def landmarks_mappling(landmarks, finger_tips):
    # rearrange index according to the standard
    new_landmarks = np.zeros((NUM_JOINTS, 3))

    new_landmarks[0] = landmarks[0]

    new_landmarks[1] = landmarks[13]
    new_landmarks[2] = landmarks[14]
    new_landmarks[3] = landmarks[15]
    new_landmarks[4] = finger_tips[0]

    new_landmarks[5] = landmarks[1]
    new_landmarks[6] = landmarks[2]
    new_landmarks[7] = landmarks[3]
    new_landmarks[8] = finger_tips[1]

    new_landmarks[9] = landmarks[4]
    new_landmarks[10] = landmarks[5]
    new_landmarks[11] = landmarks[6]
    new_landmarks[12] = finger_tips[2]

    new_landmarks[13] = landmarks[10]
    new_landmarks[14] = landmarks[11]
    new_landmarks[15] = landmarks[12]
    new_landmarks[16] = finger_tips[4]

    new_landmarks[17] = landmarks[7]
    new_landmarks[18] = landmarks[8]
    new_landmarks[19] = landmarks[9]
    new_landmarks[20] = finger_tips[3]

    return new_landmarks


def json_sort(json_dir):
    json_head = _init_save_folder()
    with open(json_dir, 'r') as f:
        json_data = json.load(f)
        images = json_data['images']
        annotations = json_data['annotations']
    for i in tqdm(range(len(images))):
        image_info = images[i]
        for j in range(len(annotations)):
            annotation_info = annotations[j]
            if image_info['id'] == annotation_info['image_id']:
                json_head['images'].append(image_info)
                json_head['annotations'].append(annotation_info)
    json_name = os.path.basename(json_dir).split('.json')[0]
    json_path = os.path.split(json_dir)[0]
    save_dir = os.path.join(json_path, json_name+'_sort.json')

    assert (json_head['images'][-1]['id'] == json_head['annotations'][-1]['image_id'])
    with open(save_dir, 'w')as f:
        json.dump(json_head, f)
