import json
import os.path
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import cv2
from tools import draw_2d_points, VideoWriter
from copy import deepcopy


def load_json_data(json_dir):
    with open(json_dir, 'r')as f:
        json_data = json.load(f)
        images = json_data['images']
        annotations = json_data['annotations']

    annotations_dict, images_dict = defaultdict(list), defaultdict(list)

    iter_num = len(images)
    for i in tqdm(range(iter_num)):
        image_info = images[i]
        annotation_info = annotations[i]
        assert (image_info['id'] == annotation_info['image_id'])

        image_id = annotation_info['image_id']
        images_dict[f'{image_id}'].append(image_info)
        annotations_dict[f'{image_id}'].append(annotation_info)

    return images_dict, annotations_dict

def get_image(image_dir, image_id, images_dict, annotations_dict, txt=''):
    if str(image_id) in images_dict.keys():
        annotation_info_list = annotations_dict[str(image_id)]
        image_info_list = images_dict[str(image_id)]

        image_dir = os.path.join(image_dir, image_info_list[0]['file_name'])
        image = cv2.imread(image_dir)

        for annotation_info, image_info in zip(annotation_info_list, image_info_list):
            bbox = convert_box(annotation_info['bbox'])
            keypoints = annotation_info['keypoints']
            id_image = image_info['id']
            id_annotation = annotation_info['id']
            image_id_annotation = annotation_info['image_id']

            info1 = f"""{id_image}"""
            info2 = f"""{id_annotation}\t{image_id_annotation}"""

            image = draw_2d_points(np.array(keypoints).reshape(21, 3), image)
            image = cv2.rectangle(image, bbox[0:2], bbox[2:4], (0, 0, 255), 2)
            image = cv2.putText(image, f"{bbox}", (bbox[0], bbox[3]-10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            image = cv2.putText(image, f"{info1}", (bbox[0], bbox[3] +40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            image = cv2.putText(image, f"{info2}", (bbox[0], bbox[3] + 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    image = cv2.putText(image, txt, (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
    return image


def convert_box(bbox):
    x_left, y_top, box_w, box_h = bbox
    x_right = x_left + box_w
    y_bottle = y_top + box_h
    return[x_left, y_top, x_right, y_bottle]


def main():
    dt_image_dir = r'E:\test_data\test_data_from_whole_body\eval_test\dt_images'
    gt_image_dir = r'E:\test_data\test_data_from_whole_body\eval_test\gt_images'
    mode = r'crop'

    gt_dir = fr'E:\test_data\test_data_from_whole_body\eval_test\annotations\average_pseudo_labels_update-coco_id-{mode}.json'
    dt_mediapipe_full_dir = fr'E:\test_data\test_data_from_whole_body\eval_test\annotations\mediapipe_full-vedio-coco_id-{mode}.json'

    print(f"loading the '{gt_dir}'......")
    gt_images_dict, gt_annotations_dict = load_json_data(gt_dir)
    print(f"loading the '{dt_mediapipe_full_dir}'......")
    dt_mediapipe_full_images_dict, dt_mediapipe_full_annotations_dict = load_json_data(dt_mediapipe_full_dir)

    image_id_list = list(gt_images_dict.keys())

    for image_id in image_id_list:
        gt_image = get_image(gt_image_dir, image_id, gt_images_dict,
                             gt_annotations_dict, 'gt')

        dt_mediapipe_full_image = get_image(dt_image_dir, image_id, dt_mediapipe_full_images_dict,
                                            dt_mediapipe_full_annotations_dict, 'mediapipe-full')
        # dt_mediapipe_full_image = cv2.resize(dt_mediapipe_full_image, (int(gt_size[1]), int(gt_size[0])),
        #                                      interpolation=cv2.INTER_CUBIC)


        canves = np.concatenate([dt_mediapipe_full_image, gt_image], axis=1)

        cv2.imshow("test", canves)
        cv2.waitKey(0)


if __name__ == '__main__':
    main()