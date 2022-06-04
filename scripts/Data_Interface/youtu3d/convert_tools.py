import os
import cv2
import numpy as np

from library.tools import draw_2d_points

DATA_CAPTURED = '2022-03-18 14:00:00'
COCOBBOX_FACTOR = 1.5
CROP_FACTOR = 2.2
MIN_SIZE = 48

BOX_FACTOR = 2.2
ROTATE = False
PARTITION = False
VALIDATION_RATE = 0.2


def crop_box(img, hand_pts_2d, box_factor=BOX_FACTOR):
    hand_min = np.min(hand_pts_2d, axis=0)
    hand_max = np.max(hand_pts_2d, axis=0)
    hand_box_c = (hand_max + hand_min) / 2.
    half_size = int(np.max(hand_max - hand_min) * box_factor / 2.)

    x_left = int(hand_box_c[0] - half_size)
    y_top = int(hand_box_c[1] - half_size)
    x_right = x_left + 2 * half_size
    y_bottom = y_top + 2 * half_size
    save_pts = hand_pts_2d - np.array([x_left, y_top])

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
    img_save[y_start:y_end, x_start:x_end] = img[y_top:y_bottom, x_left:x_right]

    return img_save, save_pts


def refine_keypts(key_pts):
    coco_kps = np.zeros((21, 3), dtype=key_pts.dtype)
    coco_kps[:, :2] = key_pts
    coco_kps[:, 2] = 2
    for idx in range(coco_kps.shape[0]):
        if coco_kps[idx, 0] <= -50 and coco_kps[idx, 1] <= -50:
            coco_kps[idx] = 0
    return coco_kps


def convert_coco_format_from_crop(file_name, image_dir, handlandmarks, head, mode, save_dir, coco_id, coco_factor=COCOBBOX_FACTOR):
    # frame_name = img_path.split('/')[-1]
    # folder_name = img_path.split('/')[-4]
    # file_name = str(img_id).zfill(12) + '.jpg'
    img = cv2.imread(image_dir)
    landmarks = handlandmarks
    img_id = coco_id
    img_path= image_dir
    json_file = head


    crop_img, crop_landmarks = crop_box(img, landmarks[:, :2].copy(), box_factor=CROP_FACTOR)
    img_h, img_w = crop_img.shape[:2]
    key_pts = crop_landmarks[:, :2].copy()  # exclude z-coordinate
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

    image_dict = dict({
        'license': 1,
        'file_name': file_name,
        'coco_url': 'Unavailable',
        'height': img_h,
        'width': img_w,
        'date_captured': DATA_CAPTURED,
        'flickr_url': 'Unavailable',
        'image_dir': image_dir,
        'id': img_id
    })

    coco_kps = coco_kps.flatten().tolist()
    anno_dict = dict({
        'segmentation': [[x_left, y_top, x_right, y_top, x_right, y_bottom, x_left, y_bottom]],
        'num_keypoints': 21,
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
    # save_path = os.path.join(save_dir, 'images', f'{mode}', file_name)

    # print(image_dir)
    # draw_img = draw_2d_points(crop_landmarks, crop_img)
    # cv2.imshow('a', draw_img)
    # cv2.waitKey(1)

    # crop_img = draw_point(key_pts, crop_img)   # show landmarks to check the landmarks correct or not
    # cv2.imwrite(save_path, crop_img)
    return 1


# img_path, landmarks, json_file, hand_id, img_id
# file_name, image_dir, handlandmarks, head, coco_id
def convert_coco_format_from_wholebody(img_id, image_dir, landmarks, json_file, coco_id,
                        save_path, mode, coco_factor=COCOBBOX_FACTOR):
    file_name = str(img_id).zfill(12) + '.jpg'
    img = cv2.imread(image_dir)

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
        print(f'TOO SMALL! img_path: {image_dir}')
        print(f'box_h: {box_h}, box_w: {box_w}\n')
        return 0

    if mode != None and save_path != None:
        save_path = os.path.join(save_path, 'images', f'{mode}2017')
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        save_dir = os.path.join(save_path, file_name)
        cv2.imshow('a', draw_2d_points(landmarks, img))
        cv2.waitKey(1)
        cv2.imwrite(save_dir, img)

    image_dict = dict({
        'license': 1,
        'file_name': file_name,
        'image_dir': image_dir,
        'coco_url': 'Unavailable',
        'height': img_h,
        'width': img_w,
        'date_captured': DATA_CAPTURED,
        'flickr_url': 'Unavailable',
        'id': coco_id
    })

    coco_kps = coco_kps.flatten().tolist()
    anno_dict = dict({
        'segmentation': [[x_left, y_top, x_right, y_top, x_right, y_bottom, x_left, y_bottom]],
        'num_keypoints': 21,
        'area': box_h * box_w,
        'iscrowd': 0,
        'keypoints': coco_kps,
        'image_id': img_id,
        'bbox': [x_left, y_top, box_w, box_h],  # 1.5 expand
        'category_id': 1,
        'id': coco_id
    })

    json_file['images'].append(image_dict)
    json_file['annotations'].append(anno_dict)

    # crop_img = draw_point(key_pts, crop_img)   # show landmarks to check the landmarks correct or not
    return 1
