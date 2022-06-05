import json
import numpy as np
import cv2
from tqdm import tqdm
from tools import draw_2d_points
from json_tools import make_json_head
from copy import deepcopy

mode = 'train'

new_json_dir = fr'E:\Data\landmarks\YouTube3D\YouTube3D-wholebody\annotations\{mode}2017-update-for-invaliddata.json'
with_left_label_json_dir1 = fr'E:\left_hand_label_data\annotations\youtu3d_update\youtu_train.json'
with_left_label_json_dir2 = fr'E:\left_hand_label_data\annotations\youtu3d_update\youtu_val.json'
save_dir = fr'E:\left_hand_label_data\annotations\youtu3d_update\{mode}2017_update.json'
debug = 0


def get_box(landmarks):
    if isinstance(landmarks, list):
        landmarks = np.array(landmarks).reshape(21, 3)
    valid_bool = landmarks[:, -1].astype(np.bool)
    key_pts = landmarks[:, :2][valid_bool]

    hand_min = np.min(key_pts, axis=0)
    hand_max = np.max(key_pts, axis=0)
    hand_box_c = (hand_max + hand_min) / 2  # (2, )
    half_size = int(np.max(hand_max - hand_min)/ 2.)

    x_left = int(hand_box_c[0] - half_size)
    y_top = int(hand_box_c[1] - half_size)
    x_right = x_left + 2 * half_size
    y_bottom = y_top + 2 * half_size
    return (x_left, y_top, x_right, y_bottom)


def bb_iou(box_a, box_b):
    box_a = np.array(box_a).flatten()
    box_b = np.array(box_b).flatten()

    # determine the (x, y)-coordinates of the intersection rectangle
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])

    # compute the area of intersection rectangle
    inter_area = abs(max((x_b - x_a, 0)) * max((y_b - y_a), 0))
    if inter_area == 0:
        return 0

    # compute the area of both the prediction and ground-truth rectangles
    box_a_area = abs((box_a[2] - box_a[0]) * (box_a[3] - box_a[1]))
    box_b_area = abs((box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))

    # compute the intersection over union by taking the intersection area and dividing it by the sum of
    # prediction + ground-truth areas - the interesection area
    iou = inter_area / float(box_a_area + box_b_area - inter_area)

    # return the intersection over union value
    return iou


def check_image(image_dir, w_keypoints, n_keypoints, debug):
    img = cv2.imread(image_dir)
    img = draw_2d_points(np.array(w_keypoints).reshape(21, 3), img)
    img = draw_2d_points(np.array(n_keypoints).reshape(21, 3), img)
    cv2.imshow('test', img)
    cv2.waitKey(~debug)

def main():
    json_head = make_json_head()
    if debug:
        cv2.namedWindow('test', cv2.WINDOW_NORMAL)
    with open(new_json_dir, 'r')as f:
        n_json_data = json.load(f)
        n_images = n_json_data['images']
        n_annotations = n_json_data['annotations']
        assert (len(n_images)==len(n_annotations))

    with open(with_left_label_json_dir1, 'r')as f:
        w_json_data1 = json.load(f)
        w_images = w_json_data1['images']
        w_annotations = w_json_data1['annotations']
        assert (len(w_images)==len(w_annotations))
    with open(with_left_label_json_dir2, 'r')as f:
        w_json_data2 = json.load(f)
        w_images.extend(w_json_data2['images'])
        w_annotations.extend(w_json_data2['annotations'])
        assert (len(w_images)==len(w_annotations))

    w_iter_num = len(w_images)
    n_iter_num = len(n_images)

    for i in tqdm(range(n_iter_num)):
        n_image_info = n_images[i]
        n_annotation_info = n_annotations[i]

        n_image_dir = n_image_info['image_dir']
        n_keypoints = n_annotation_info['keypoints']
        file_name = n_image_info['file_name']

        for j in range(w_iter_num):
            w_image_info = w_images[j]
            w_annotation_info = w_annotations[j]

            w_image_dir = w_image_info['image_dir']
            w_keypoints = w_annotation_info['keypoints']

            if n_image_dir == w_image_dir:
                n_box = get_box(n_keypoints)
                w_box = get_box(w_keypoints)
                iou = bb_iou(n_box, w_box)

                # print(f"file_name:{file_name}\tiou:{iou}")
                # check_image(n_image_dir, w_keypoints, n_keypoints, debug)

                if iou==1:
                    image_info = deepcopy(w_image_info)
                    annotation_info = deepcopy(w_annotation_info)

                    image_info['id'] = n_image_info['id']
                    image_info['file_name'] = n_image_info['file_name']
                    annotation_info['id'] = n_annotation_info['id']
                    annotation_info['image_id'] = n_annotation_info['image_id']

                    json_head['images'].append(image_info)
                    json_head['annotations'].append(annotation_info)

    with open(save_dir, 'w')as f:
        json.dump(json_head, f)
        print(f"Success to write {save_dir}")


if __name__ == '__main__':
    main()