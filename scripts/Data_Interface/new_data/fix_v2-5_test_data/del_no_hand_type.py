"""
问题：利用之前带左右手标签的v2_5json对新的v2_5json文件赋值手标签，会有部分对象没有手标签
该程序目的就是根据del_list中的信息删除对应图片中没有手标签的数据
"""
import json
from collections import defaultdict

from json_tools import make_json_head

def main():
    del_list = [1400045, 1400212, 1400333, 1401267, 1401504, 1402210, 1402536, 1403577, 1403976]
    # [1400333, 1401267, 1401504, 1402536, 1403577, 1403976]

    json_path = r'G:\test_data\new_data\new_data_from_whole_body\annotations\test_w_hand_type.json'
    images_dict, annotations_dict = load_json_data(json_path)

    del_list.sort()
    del_list = del_list[::-1]

    for del_index in del_list:
        record = []
        annotations_info_list = annotations_dict[del_index]
        for index, annotations_info in enumerate(annotations_info_list):
            if "hand_type" not in annotations_info.keys():
                record.append(index)
        record.sort()
        record = record[::-1]
        for i in record:
            del annotations_info_list[i]
        if len(annotations_info_list) == 0:
            annotations_dict.pop(del_index)
            images_dict.pop(del_index)

    write_json(images_dict, annotations_dict, json_path)


def get_ids(data_dict):
    return list(data_dict.keys())


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


if __name__=="__main__":
    main()