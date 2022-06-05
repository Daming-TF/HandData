import os
from collections import defaultdict
import json
from tqdm import tqdm
import numpy as np

from library.json_tools import make_json_head


def main():
    # w_hand_json_dir = r'E:\left_hand_label_data\annotations'
    # w_hand_json_dir = r'E:\left_hand_label_data\annotations\person_keypoints_val2017.json'
    w_hand_json_path = r'E:\left_hand_label_data\annotations\v2_6\remark_no_hand\2022-4-20-17-0-57.json'
    # G:\transmission\anno\person_keypoints_val2017.json
    new_json_path = r'E:\left_hand_label_data\annotations\v2_6\person_keypoints_val2017.json'
    save_path = r'E:\left_hand_label_data\annotations\v2_6\remark_no_hand\wh_val2017.json'

    # wh_images_dict, wh_annotations_dict = load_data(w_hand_json_dir)
    # wh_images_dict, wh_annotations_dict = wh_load_json_data(w_hand_json_dir)
    wh_images_dict, wh_annotations_dict = new_load_json_data(w_hand_json_path)
    new_images_dict, new_annotations_dict = new_load_json_data(new_json_path)

    for image_path in tqdm(get_image_paths(new_images_dict)):
        new_annotation_unit_list = new_annotations_dict[image_path]
        wh_annotation_unit_list = wh_annotations_dict[image_path]

        # # debug
        # if not image_path == r'G:\\imgdate2\\HO3D_v3\\HO3D_v3\\train\\GSF14\\rgb\\0678.jpg'.replace(r'\\', '\\'):
        #     continue
        # print(image_path)
        # for new_annotation_unit in new_annotation_unit_list:
        #     new = np.array(new_annotation_unit['keypoints']).reshape(21, 3).astype(int)
        #     new_valid_bool = new[:, -1].astype(bool)
        #     print(new.tolist())
        #     print("---------------------------------")
        #     for wh_annotation_unit in wh_annotation_unit_list:
        #         wh = np.array(wh_annotation_unit['keypoints']).reshape(21, 3).astype(int)
        #         print(wh.tolist())
        #         wh_valid_bool = wh[:, -1].astype(bool)
        #         valid_bool = new_valid_bool & wh_valid_bool
        #         print(valid_bool)
        #         print(f"sum:{np.sum(np.abs(new[:,:2][valid_bool]-wh[:,:2][valid_bool]))}")
        #         # debug

        if not (len(new_annotation_unit_list) != 0 and len(wh_annotation_unit_list) != 0):
            continue
        match_dict = match_hand(new_annotation_unit_list, wh_annotation_unit_list, wh_images_dict[image_path])

        for index in match_dict.keys():
            new_annotation_unit_list[index]['hand_type'] = wh_annotation_unit_list[match_dict[index]]['hand_type']

    # save_name = os.path.basename(new_json_path)
    # save_path = os.path.join(save_dir, save_name)
    write_json(new_images_dict, new_annotations_dict, save_path)


def write_json(images_dict, annotations_dict, save_path):
    json_head = make_json_head()
    image_paths = get_image_paths(images_dict)
    for image_path in image_paths:
        image_info = images_dict[image_path]
        json_head["images"].append(image_info)

        annotation_info_list = annotations_dict[image_path]
        for annotation_info in annotation_info_list:
            json_head["annotations"].append(annotation_info)

    with open(save_path, 'w')as f:
        json.dump(json_head, f)


def match_hand(new_annotation_list, wh_annotation_list, wh_images_unit):
    match_dict = {}
    for index, new_annotation_unit in enumerate(new_annotation_list):
        new_keypoints = np.array(new_annotation_unit['keypoints']).reshape(21, 3)
        new_keypoints[:, :2] = new_keypoints[:, :2].astype(int)
        new_valid_bool = new_keypoints[:, -1].astype(bool)
        for match_index, wh_annotation_unit in enumerate(wh_annotation_list):
            wh_keypoints = np.array(wh_annotation_unit['keypoints']).reshape(21, 3)
            wh_keypoints[:, :2] = wh_keypoints[:, :2].astype(int)
            wh_valid_bool = wh_keypoints[:, -1].astype(bool)
            valid_bool = new_valid_bool & wh_valid_bool
            if np.sum(np.abs(new_keypoints[:, :2][valid_bool]-wh_keypoints[:, :2][valid_bool])) < 4:
                match_dict[index] = match_index

    return match_dict


def wh_load_json_data(json_dir):
    annotations_dict = defaultdict(list)
    images_dict, images_path_dict = {}, {}

    mode_list = ['val', 'train']
    for mode in mode_list:
        json_path = os.path.join(json_dir, f'person_keypoints_{mode}2017.json')

        print(f"loading the json file {json_path}......")
        with open(json_path, 'r') as f:
            dataset = json.load(f)

        for img in dataset['images']:
            images_path_dict[img['id']] = img['image_dir']

        for img in dataset['images']:
            hand_id = img['id']
            images_dict[images_path_dict[hand_id]] = img

        for ann in dataset['annotations']:
            hand_id = ann['id']
            annotations_dict[images_path_dict[hand_id]].append(ann)

    return images_dict, annotations_dict


def new_load_json_data(json_path):
    annotations_dict = defaultdict(list)
    images_dict, images_path_dict = {}, {}
    counter = 0
    with open(json_path, 'r') as f:
        dataset = json.load(f)

    for img in dataset['images']:
        images_path_dict[img['id']] = img['image_dir']

    for img in dataset['images']:
        image_id = img['id']
        images_dict[images_path_dict[image_id]] = img

    for ann in dataset['annotations']:
        image_id = ann['image_id']
        if image_id not in images_path_dict.keys():
            print(image_id)
            continue
        annotations_dict[images_path_dict[image_id]].append(ann)
        counter += 1
    print(counter)

    return images_dict, annotations_dict


def get_image_paths(images_dict):
    return list(images_dict.keys())


if __name__ == '__main__':
    main()