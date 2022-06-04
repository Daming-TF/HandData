import os
import cv2
import shutil
import numpy as np

COCOBBOX_FACTOR = 1.5
CROP_FACTOR = 2.2
MIN_SIZE = 48
DATA_CAPTURED = '2021-9-26 14:52:28'
NUM_HAND_KEYPOINTS = 21


def convert_coco_format_for_crop(json_file, image_info, annotation_info, data_dir, save_dir, image_id,
                                 is_gt=False, gt_box=[], save_flag=1):
    # json_crop, image_info, annotation_in
    landmarks = np.array(annotation_info["keypoints"]).reshape(21, 3)
    file_name = image_info["file_name"]

    image_path = os.path.join(data_dir, file_name)
    img = cv2.imread(image_path)

    # 规范图片名
    file_name = str(image_id).zfill(12) + '.jpg'

    if np.all(landmarks==0):
        return 0

    crop_img, crop_landmarks, crop_coordinate_system_mark = crop_box(img, landmarks.copy(), box_factor=CROP_FACTOR,
                                                                     is_gt=is_gt, gt_box=gt_box)
    img_h, img_w = crop_img.shape[:2]
    assert isinstance(image_id, int)

    # *** add 8.26 ***
    # 重塑landmarks，把第1,2列换为以crop为边界的crop_landmarks,把第三列数值全转化为0
    coco_kps = landmarks.copy()
    coco_kps[:, :2] = crop_landmarks[:, :2]
    coco_kps[:, 2] = landmarks[:, 2]
    kps_valid_bool = coco_kps[:, -1].astype(np.bool)
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
        'file_name': file_name,
        'coco_url': 'Unavailable',
        'height': img_h,
        'width': img_w,
        'date_captured': DATA_CAPTURED,
        'flickr_url': 'Unavailable',
        'id': image_id
    })

    coco_kps = coco_kps.flatten().tolist()
    # segmentation分别表示左上，右上，右下，左下四个点坐标
    anno_dict = dict({
        'segmentation': [[x_left, y_top, x_right, y_top, x_right, y_bottom, x_left, y_bottom]],
        'num_keypoints': NUM_HAND_KEYPOINTS,
        'area': box_h * box_w,
        'iscrowd': 0,
        'keypoints': coco_kps,
        'image_id': image_id,
        'bbox': [x_left, y_top, box_w, box_h],  # 1.5 expand
        'category_id': 1,
        'id': image_id,
        'score':1
    })

    if save_flag:
        json_file['images'].append(image_dict)
    json_file['annotations'].append(anno_dict)
    save_path = os.path.join(save_dir, file_name)

    # crop_img = draw_landmark(np.asarray(coco_kps).reshape(NUM_HAND_KEYPOINTS, 3), hand_type, crop_img)
    cv2.imwrite(save_path, crop_img)

    return crop_coordinate_system_mark


def crop_box(img, hand_pts_2d, box_factor=2.2, is_gt=False, gt_box=[]):
    record_box = []
    if is_gt:
        coco_kps = hand_pts_2d.copy()
        # 将landmorks的第三列转化为0/1
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

        record_box = [x_left, y_top, x_right, y_bottom]

    else:
        [x_left, y_top, x_right, y_bottom] = gt_box
        half_size = int((x_right - x_left)/2)

    # 纠正坐标
    save_pts = hand_pts_2d[:, :2] - np.array([x_left, y_top])

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
        return img, save_pts
    img_save[y_start:y_end, x_start:x_end] = img[y_top:y_bottom, x_left:x_right]

    return img_save, save_pts, record_box


def convert_coco_format_for_whole(json_file, image_info, annotation_info,  data_dir, save_dir, save_flag=1):
    file_name = image_info["file_name"]
    data_path = os.path.join(data_dir, file_name)
    save_path = os.path.join(save_dir, file_name)
    shutil.copyfile(data_path, save_path)

    if save_flag:
        json_file["images"].append(image_info)
    json_file["annotations"].append(annotation_info)
