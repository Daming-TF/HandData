"""
把v2_5数据加入左右手标签，保存成json
缺点：
可能有部分手是没有左右手标签的，原因是：当时边左右手的时候，发现有部分手的坐标点是不对的
"""

import json
from collections import defaultdict

from tools import bb_iou
from json_tools import make_json_head

w_lhand_json_path = r'E:\left_hand_label_data\annotations\person_keypoints_test2017.json'
new_json_path = r'G:\test_data\new_data\new_data_from_whole_body\annotations\test-update.json'
save_path = r'G:\test_data\new_data\new_data_from_whole_body\annotations\test_w_hand_type.json'


def main():
    w_lhand_images_dict, w_lhand_annotations_dict = load_json_data(w_lhand_json_path)
    new_images_dict, new_annotations_dict = load_json_data(new_json_path)

    w_lhand_ids = get_ids(w_lhand_images_dict)
    new_ids = get_ids(new_annotations_dict)

    for w_lhand_id in w_lhand_ids:
        print(w_lhand_id)
        # if not(w_lhand_id == 1401504):
        #     continue
        w_lhand_image_info = w_lhand_images_dict[w_lhand_id]
        w_lhand_image_dir = w_lhand_image_info["image_dir"]

        for new_id in new_ids:
            new_image_info = new_images_dict[new_id]
            new_image_dir = new_image_info["image_dir"]

            if w_lhand_image_dir == new_image_dir:
                w_lhand_annotation_info_list = w_lhand_annotations_dict[w_lhand_id]
                new_annotations_info_list = new_annotations_dict[new_id]
                match_dict = find_match(new_annotations_info_list, w_lhand_annotation_info_list)
                add_hand_label(new_annotations_info_list, w_lhand_annotation_info_list, match_dict)

    write_json(new_images_dict, new_annotations_dict, save_path)


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


def add_hand_label(annotations_info1_list, annotations_info2_list, match_dict):
    for index, annotations_info1 in enumerate(annotations_info1_list):
        if index in match_dict.keys():
            match_id = match_dict[index]
            annotations_info1_list[index]["hand_type"] = annotations_info2_list[match_id]["hand_type"]
        # else:
        #     annotations_info1_list[index]["hand_type"] = "right"


def find_match(annotations_info1_list, annotations_info2_list):
    match_dict = {}
    for new_index, annotations_info1 in enumerate(annotations_info1_list):
        max_iou_index = None
        bbox1 = annotations_info1["bbox"]
        for w_lhand_index, annotations_info2 in enumerate(annotations_info2_list):
            bbox2 = annotations_info2["bbox"]
            iou = bb_iou(convert_box(bbox1), convert_box(bbox2))
            if iou == 1:
                max_iou_index = w_lhand_index

        if max_iou_index is not None:
            match_dict[new_index] = w_lhand_index

    return match_dict


def convert_box(bbox):
    x_left, y_top, box_w, box_h = bbox
    return [x_left, y_top, x_left+box_w, y_top+box_h]

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


def get_ids(data_dict):
    return list(data_dict.keys())


if __name__ == '__main__':
    main()