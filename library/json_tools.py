import os
import numpy as np
import cv2
import json
from collections import defaultdict


def _init_save_folder(save_dir):
    # 创建数据集路径
    os.makedirs(os.path.join(save_dir, 'images', 'train2017'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'images', 'val2017'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'images', 'test2017'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'annotations'), exist_ok=True)
    # 向json文件写入作者等相关信息
    json_file = make_json_head()
    return json_file


def make_json_head():
    '''
    编写json文件头的元素，包括作者信息，格式等，并返回coco格式规范的

    Returns
    -------
    一个符合coco格式的字典对象

    '''
    json_struct = dict()

    json_struct['info'] = dict()
    json_struct['info']['description'] = r'The hand keypoint dataset of HUYA.'
    json_struct['info']['url'] = r'Unavailable'
    json_struct['info']['version'] = r'2.0'
    json_struct['info']['year'] = 2021
    json_struct['info']['contributor'] = r'Vision Team of HUYA AI Center'
    json_struct['info']['date_created'] = r'2021-09-21 15:47:40'

    json_struct['licenses'] = list()
    json_struct['licenses'].append(dict({'url': 'Unavailable', 'id': 1, 'name': 'HUYA License'}))

    json_struct['categories'] = list()
    json_struct['categories'].append(dict({
        'supercategory': 'hand',
        'id': 1,
        'name': 'hand',
        'keypoints': [
            'wrist',  # 1
            'thumb1', 'thumb2', 'thumb3', 'thumb4',  # 2-5
            'index1', 'index2', 'index3', 'index4',  # 6-9
            'middle1', 'middle2', 'middle3', 'middle4',  # 10-13
            'ring1', 'ring2', 'ring3', 'ring4',  # 14-17
            'pinky1', 'pinky2', 'pinky3', 'pinky4'  # 18-21
        ],
        'skeleton': [
            [1, 2], [2, 3], [3, 4], [4, 5],
            [1, 6], [6, 7], [7, 8], [8, 9],
            [1, 10], [10, 11], [11, 12], [12, 13],
            [1, 14], [14, 15], [15, 16], [16, 17],
            [1, 18], [18, 19], [19, 20], [20, 21],
            [6, 10], [10, 14], [14, 18]
        ]
    }))

    json_struct['images'] = list()
    json_struct['annotations'] = list()
    return json_struct


BOX_FACTOR = 2.2


def crop_box(img, hand_pts_2d, box_factor=BOX_FACTOR):
    '''

    Parameters
    ----------
    img : Original image
    hand_pts_2d : This is a 21*2 matrix
    box_factor : 缩放因子

    Returns
    -------
    img_save : 剪辑好的照片
    save_pts ： 基于剪辑后的坐标系的landmarks

    '''
    # *** add 8.26 ***
    coco_kps = hand_pts_2d.copy()
    # 将landmorks的第三列转化为0/1
    kps_valid_bool = coco_kps[:, -1].astype(np.bool)
    new_hand_pts_2d = coco_kps[:, :2][kps_valid_bool]
    # *** add 8.26 ***

    # 计算新的额边框边界
    hand_min = np.min(new_hand_pts_2d, axis=0)
    hand_max = np.max(new_hand_pts_2d, axis=0)
    hand_box_c = (hand_max + hand_min) / 2.
    half_size = int(np.max(hand_max - hand_min) * box_factor / 2.)

    x_left = int(hand_box_c[0] - half_size)
    y_top = int(hand_box_c[1] - half_size)
    x_right = x_left + 2 * half_size
    y_bottom = y_top + 2 * half_size
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


    return img_save, save_pts

COCOBBOX_FACTOR = 1.5
CROP_FACTOR = 2.2
MIN_SIZE = 48
DATA_CAPTURED = '2021-9-26 14:52:28'
NUM_HAND_KEYPOINTS = 21

def convert_coco_format(img, landmarks, json_file, mode, save_dir, img_id):
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
        'id': img_id
    })

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
    save_path = os.path.join(save_dir, 'images', f'{mode}', file_name)

    # crop_img = draw_landmark(np.asarray(coco_kps).reshape(NUM_HAND_KEYPOINTS, 3), hand_type, crop_img)
    cv2.imwrite(save_path, crop_img)

    return 1


def load_json_data(json_path):
    annotations_dict = defaultdict(list)
    images_dict = {}
    with open(json_path, 'r')as f:
        dataset = json.load(f)

    for ann in dataset['annotations']:
        annotations_dict[ann['image_id']].append(ann)

    for img in dataset['images']:
        images_dict[img['id']] = img

    return images_dict, annotations_dict


def write_json(images_dict, annotations_dict, save_path):
    json_head = make_json_head()
    images_ids = get_ids(images_dict)
    for image_id in images_ids:
        image_info = images_dict[image_id]
        json_head["images"].append(image_info)

        annotation_info_list = annotations_dict[image_id]
        for annotation_info in annotation_info_list:
            json_head["annotations"].append(annotation_info)

    with open(save_path, 'w')as f:
        json.dump(json_head, f)


def get_ids(data_dict):
    return list(data_dict.keys())


def convert_landmarks(bbox, i_h, i_w):      # box:left_x, y_top, h, w
    t_x, t_y = int(bbox[0] + bbox[3] / 2), int(bbox[1] + bbox[2] - 20)   # box bottom middle
    while 1:
        if t_x-10 < 0 or t_x+10 > i_w:
            t_x = t_x + 10 if t_x-10 < 0 else t_x - 10
        elif t_y-10 < 0 or t_y+10 > i_h:
            t_y = t_y + 10 if t_y-10 < 0 else t_y - 10
        else:
            break

    return t_x, t_y
