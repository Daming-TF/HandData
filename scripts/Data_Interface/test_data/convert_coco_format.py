import cv2
import os
import numpy as np
from tools import draw_2d_points

COCOBBOX_FACTOR = 1.5
CROP_FACTOR = 2.2
MIN_SIZE = 48
DATA_CAPTURED = '2021-9-26 14:52:28'

def refine_keypts(key_pts):
    coco_kps = np.zeros((key_pts.shape[0], 3), dtype=key_pts.dtype)
    coco_kps[:, :2] = key_pts
    coco_kps[:, 2] = 2
    for idx in range(coco_kps.shape[0]):
        if coco_kps[idx, 0] <= -50 and coco_kps[idx, 1] <= -50:
            coco_kps[idx] = 0
        if np.all(coco_kps[idx, :2] == 0):
            coco_kps[idx, 2] = 0
    return coco_kps

# def convert_coco_format_from_wholebody(img_path, landmarks, json_file, mode=None, save_dir=None, coco_factor=COCOBBOX_FACTOR):
#     # frame_name = img_path.split('/')[-1]
#     # folder_name = img_path.split('/')[-4]
#     img = cv2.imread(img_path)
#     file_name = os.path.basename(img_path)
#     img_id = int(file_name.split('.')[0])
#
#     img_h, img_w = img.shape[:2]
#     key_pts = landmarks[:, :2].copy()  # exclude z-coordinate
#     assert isinstance(img_id, int)
#
#     # 计算坐标
#     coco_kps_list = [[], []]
#     coco_kps_l = landmarks[0:21]
#     coco_kps_list[0] = coco_kps_l.flatten().tolist()
#     coco_kps_l = landmarks[21:42]
#     coco_kps_list[1] = coco_kps_l.flatten().tolist()
#
#
#     if not np.all(landmarks == 0):
#         # *** add 8.26 ***
#         coco_kps = refine_keypts(key_pts)
#         kps_valid_bool = coco_kps[:, -1].astype(np.bool)
#         key_pts = coco_kps[:, :2][kps_valid_bool]
#         # *** add 8.26 ***
#
#         hand_min = np.min(key_pts, axis=0)      # (2,)
#         hand_max = np.max(key_pts, axis=0)      # (2,)
#         hand_box_c = (hand_max + hand_min) / 2  # (2, )
#         half_size = int(np.max(hand_max - hand_min) * coco_factor / 2.)  # int
#
#         x_left = int(hand_box_c[0] - half_size)
#         y_top = int(hand_box_c[1] - half_size)
#         x_right = x_left + 2 * half_size
#         y_bottom = y_top + 2 * half_size
#         box_w = x_right - x_left
#         box_h = y_bottom - y_top
#         if min(box_h, box_w) < MIN_SIZE:
#             print(f'TOO SMALL! img_path: {img_path}')
#             print(f'box_h: {box_h}, box_w: {box_w}\n')
#             return 0
#
#     else:
#         x_left, y_top, x_right, y_bottom = 0, 0, 0, 0
#         box_h, box_w =0, 0
#
#     # save_path = os.path.join(save_dir, 'images', f'{mode}', file_name)
#
#     image_dict = dict({
#         'license': 1,
#         'file_name': file_name,
#         'image_dir': img_path,
#         'coco_url': 'Unavailable',
#         'height': img_h,
#         'width': img_w,
#         'date_captured': DATA_CAPTURED,
#         'flickr_url': 'Unavailable',
#         'id': img_id
#     })
#
#
#     anno_dict = dict({
#         'segmentation': [[x_left, y_top, x_right, y_top, x_right, y_bottom, x_left, y_bottom]],
#         'num_keypoints': 21,
#         'area': box_h * box_w,
#         'iscrowd': 0,
#         'keypoints': coco_kps_list,
#         'image_id': img_id,
#         'bbox': [x_left, y_top, box_w, box_h],  # 1.5 expand
#         'category_id': 1,
#         'id': img_id
#     })
#
#     for landmarks in coco_kps_list:
#         if len(landmarks) == 0:
#             continue
#         landmarks = np.array(landmarks).reshape(21, 3)
#         img = draw_2d_points(landmarks, img)
#     # cv2.imshow('Window', img)
#     # cv2.waitKey(1)
#
#     json_file['images'].append(image_dict)
#     json_file['annotations'].append(anno_dict)
#
#     # crop_img = draw_point(key_pts, crop_img)   # show landmarks to check the landmarks correct or not
#     # cv2.imshow('a', draw_2d_points(landmarks, img))
#     # cv2.waitKey(0)
#     # cv2.imwrite(save_path, img)
#     return img


def convert_coco_format_from_wholebody(img, landmarks, json_file, id, data_path, coco_factor=COCOBBOX_FACTOR):
    # frame_name = img_path.split('/')[-1]
    # folder_name = img_path.split('/')[-4]

    img_id = id
    file_name = str(id).zfill(12) + '.jpg'
    image_dir = os.path.join(data_path, file_name)

    img_h, img_w = img.shape[:2]
    key_pts = landmarks[:, :2].copy()  # exclude z-coordinate
    assert isinstance(img_id, int)

    # 计算坐标
    coco_kps_list = [[], []]
    coco_kps_l = landmarks[0:21]
    coco_kps_list[0] = coco_kps_l.flatten().tolist()
    coco_kps_l = landmarks[21:42]
    coco_kps_list[1] = coco_kps_l.flatten().tolist()


    if not np.all(landmarks == 0):
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
            print(f'TOO SMALL! img_path: {img_id}')
            print(f'box_h: {box_h}, box_w: {box_w}\n')
            return 0

    else:
        x_left, y_top, x_right, y_bottom = 0, 0, 0, 0
        box_h, box_w =0, 0

    # save_path = os.path.join(save_dir, 'images', f'{mode}', file_name)

    image_dict = dict({
        'license': 1,
        'file_name': file_name,
        'image_dir': image_dir,
        'coco_url': 'Unavailable',
        'height': img_h,
        'width': img_w,
        'date_captured': DATA_CAPTURED,
        'flickr_url': 'Unavailable',
        'id': img_id
    })

    anno_dict = dict({
        'segmentation': [[x_left, y_top, x_right, y_top, x_right, y_bottom, x_left, y_bottom]],
        'num_keypoints': 21,
        'area': box_h * box_w,
        'iscrowd': 0,
        'keypoints': coco_kps_list,
        'image_id': img_id,
        'bbox': [x_left, y_top, box_w, box_h],  # 1.5 expand
        'category_id': 1,
        'score': 0.5,
        'id': img_id
    })

    for i, landmarks in enumerate(coco_kps_list):
        if len(landmarks) == 0:
            continue
        landmarks = np.array(landmarks).reshape(21, 3)
        x, y = np.max(landmarks[:, 0:2], axis=0)
        img = draw_2d_points(landmarks, img)
        if i:
            label = 'Right'
        else:
            label = 'Left'
        if not np.all(landmarks == 0):
            img = cv2.putText(img, f'{label}', (int(x),int(y)), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 3)


    json_file['images'].append(image_dict)
    json_file['annotations'].append(anno_dict)

    return img