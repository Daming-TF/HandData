import numpy as np
import cv2
import os

COCOBBOX_FACTOR = 1.5
CROP_FACTOR = 2.2
MIN_SIZE = 48
DATA_CAPTURED = '2022-04-15'
NUM_JOINTS = 21

def refine_keypts(coco_kps, img_h, img_w):
    for idx in range(coco_kps.shape[0]):
        if coco_kps[idx, 2] not in [0, 1, 2]:
            coco_kps[idx, 2] = 1
        if coco_kps[idx, 0] < 0 or coco_kps[idx, 0] >img_w:
            coco_kps[idx] = 0
        if coco_kps[idx, 1] < 0 or coco_kps[idx, 1] >img_h:
            coco_kps[idx] = 0
        if np.all(coco_kps[idx, :2] == 0):
            coco_kps[idx, 2] = 0
    return coco_kps


def convert_coco_format_from_wholebody(img_path, landmarks, json_file, mode, save_dir, coco_id, image_id,
                                       coco_factor=COCOBBOX_FACTOR, hand_type=None, save_flag=0):
    img = cv2.imread(img_path)
    file_name = str(image_id).zfill(12) + '.jpg'

    img_h, img_w = img.shape[:2]
    key_pts = landmarks[:, :2].copy()  # exclude z-coordinate
    assert isinstance(coco_id, int)

    # Check whether the key points are standardized
    coco_kps = refine_keypts(landmarks, img_h, img_w)
    if np.all(coco_kps == 0):
        return 0

    # Get the bbox info
    kps_valid_bool = coco_kps[:, -1].astype(np.bool)
    key_pts = coco_kps[:, :2][kps_valid_bool]

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
        print("\n")
        print(f'[!] TOO SMALL! img_path: {img_path} \t box_h: {box_h}, box_w: {box_w}')
        return 0

    image_dict = dict({
        'license': 1,
        'file_name': file_name,
        'image_dir': img_path,
        'coco_url': 'Unavailable',
        'height': img_h,
        'width': img_w,
        'date_captured': DATA_CAPTURED,
        'flickr_url': 'Unavailable',
        'id': image_id
    })

    coco_kps = np.around(coco_kps, 2).flatten().tolist()

    anno_dict = dict({
        'segmentation': [[x_left, y_top, x_right, y_top, x_right, y_bottom, x_left, y_bottom]],
        'num_keypoints': NUM_JOINTS,
        'area': box_h * box_w,
        'iscrowd': 0,
        'keypoints': coco_kps,
        'image_id': image_id,
        'bbox': [x_left, y_top, box_w, box_h],  # 1.5 expand
        'category_id': 1,
        'id': coco_id,
        'hand_type': hand_type
    })

    if save_flag == 0:
        json_file['images'].append(image_dict)
    json_file['annotations'].append(anno_dict)

    # Image save path
    save_path = os.path.join(save_dir, 'images', f'{mode}', file_name)
    cv2.imwrite(save_path, img)

    return 1