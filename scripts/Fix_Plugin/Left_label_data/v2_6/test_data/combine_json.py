from collections import defaultdict
import json

from json_tools import load_json_data, write_json


def main():
    json_path1 = r'E:\left_hand_label_data\annotations\v2_6\test_data_add_hand\mediapipe-detecter\test.json'
    json_path2 = r'E:\left_hand_label_data\annotations\v2_6\test_data_add_hand\manual_tagging\2022-4-24-21-14-15.json'
    save_path = r'E:\left_hand_label_data\annotations\v2_6\test_data_add_hand\person_keypoints_test2017.json'

    images_dict, annotations_dict = load_json_data(json_path1, json_path2)

    for image_id in list(images_dict.keys()):
        image_unit = images_dict[image_id]
        anno_unit_list = annotations_dict[image_id]

        for index, anno_unit in enumerate(anno_unit_list):
            handtype = anno_unit['hand_type']
            if handtype is None:
                del anno_unit_list[index]

        if len(anno_unit_list) == 0:
            images_dict.pop(image_id)
            annotations_dict.pop(image_id)

    write_json(images_dict, annotations_dict, save_path)



def load_json_data(json_path1, json_path2):
    annotations_dict = defaultdict(list)
    images_dict = {}

    for json_path in [json_path1, json_path2]:
        with open(json_path, 'r')as f:
            dataset = json.load(f)

        for ann in dataset['annotations']:
            annotations_dict[ann['image_id']].append(ann)

        for img in dataset['images']:
            images_dict[img['id']] = img

        return images_dict, annotations_dict


if __name__ == '__main__':
    main()