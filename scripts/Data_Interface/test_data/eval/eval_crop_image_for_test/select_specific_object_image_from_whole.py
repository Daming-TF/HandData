import json
import os
from collections import defaultdict
import numpy as np
from copy import deepcopy

from json_tools import make_json_head
from convert_tools import convert_coco_format_for_whole, convert_coco_format_for_crop

# mediapipe_full-vedio-coco_id.json, average_pseudo_labels_update-coco_id.json
gt_json_path = r'E:\test_data\test_data_from_whole_body\annotations\average_pseudo_labels_update-coco_id.json'
dt_json_path = r'E:\test_data\test_data_from_whole_body\annotations\mediapipe_full-vedio-coco_id.json'
data_dir= r'E:\test_data\test_data_from_whole_body\images'
gt_image_save_dir = r'E:\test_data\test_data_from_whole_body\eval_test\gt_images'
dt_image_save_dir = r'E:\test_data\test_data_from_whole_body\eval_test\dt_images'
json_save_dir = r'E:\test_data\test_data_from_whole_body\eval_test\annotations'


def load_json(json_path):
    with open(json_path, 'r')as f:
        json_data = json.load(f)
        images = json_data["images"]
        annotations = json_data["annotations"]
    return images, annotations


def main():
    json_head = make_json_head()
    gt_json_whole, gt_json_crop = deepcopy(json_head), deepcopy(json_head)
    dt_json_whole, dt_json_crop = deepcopy(json_head), deepcopy(json_head)

    crop_coordinate_system_dict = defaultdict(list)

    gt_images, gt_annotations = load_json(gt_json_path)
    dt_images, dt_annotations = load_json(dt_json_path)

    image_ids = list(np.arange(1400276, 1400286))
    # image_ids = list(np.arange(1400119, 1402070))
    for index, image_id in enumerate(image_ids):
        save_flag = 1
        hand_counter = 0
        box_id = 0
        for image_info, annotation_info in zip(gt_images, gt_annotations):
            gt_image_id_from_json = int(annotation_info["image_id"])
            if image_id == gt_image_id_from_json:
                box_id = (index+1)*2+hand_counter
                if hand_counter:
                    save_flag = 0
                convert_coco_format_for_whole(gt_json_whole, image_info, annotation_info,  data_dir, gt_image_save_dir, save_flag)
                gt_box = convert_coco_format_for_crop(gt_json_crop, image_info, annotation_info, data_dir,
                                                        gt_image_save_dir, image_id=box_id, is_gt=True, save_flag=save_flag)

                crop_coordinate_system_dict[f'{box_id}'] = gt_box     # [x_left, y_top, x_right, y_bottom]
                hand_counter += 1
                if hand_counter == 2:
                    continue

        save_flag = 1
        hand_counter = 0
        for image_info, annotation_info in zip(dt_images, dt_annotations):
            dt_image_id_from_json = int(annotation_info["image_id"])
            if image_id == dt_image_id_from_json:
                box_id = (index+1)*2+hand_counter
                if hand_counter:
                    save_flag = 0
                convert_coco_format_for_whole(dt_json_whole, image_info, annotation_info,  data_dir, dt_image_save_dir, save_flag)

                gt_box = crop_coordinate_system_dict[f'{box_id}']
                convert_coco_format_for_crop(dt_json_crop, image_info, annotation_info, data_dir,
                                                        dt_image_save_dir, image_id=box_id, is_gt=False, gt_box=gt_box, save_flag=save_flag)

                hand_counter += 1
                if hand_counter == 2:
                    continue

    json_name = os.path.basename(gt_json_path).split(".json")[0]
    whole_save_path = os.path.join(json_save_dir, json_name+'-whole.json')
    with open(whole_save_path, 'w')as wf:
        json.dump(gt_json_whole, wf)
    crop_save_path = os.path.join(json_save_dir, json_name + '-crop.json')
    with open(crop_save_path, 'w') as cf:
        json.dump(gt_json_crop, cf)

    json_name = os.path.basename(dt_json_path).split(".json")[0]
    whole_save_path = os.path.join(json_save_dir, json_name + '-whole.json')
    with open(whole_save_path, 'w') as wf:
        json.dump(dt_json_whole, wf)
    crop_save_path = os.path.join(json_save_dir, json_name + '-crop.json')
    with open(crop_save_path, 'w') as cf:
        json.dump(dt_json_crop, cf)


if __name__ == '__main__':
    main()