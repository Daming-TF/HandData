import os.path
from copy import deepcopy

from json_tools import load_json_data, write_json

def main(ignore_path, json_path, save_path):
    with open(ignore_path, "r") as f:  # 打开文件
        ignore_info_list = list()
        for ignore_info in f.readlines():
            ignore_info = ignore_info.strip('\n')
            image_path = ignore_info.split('**')[0]
            image_id = int(os.path.basename(image_path).split('.jpg')[0])
            ignore_info_list.append(image_id)

    # json文件结构(字典结构)包含：'info'-dict, 'licenses'-list(dict),
    # 'categories'-list(dict), 'images'-list(dict), 'annotations'-list(dict)
    images_dict, annotations_dict = load_json_data(json_path)

    image_ids = deepcopy(list(images_dict.keys()))

    for image_id in image_ids:
        if image_id in ignore_info_list:
            images_dict.pop(image_id)
            annotations_dict.pop(image_id)

    write_json(images_dict, annotations_dict, save_path)



if __name__ == "__main__":
    ignore_path_ = \
        r"G:\test_data\new_data\new_data_from_whole_body_v2_6\weed_out_badcase\v2_6-badcase.txt"
    json_path_ = \
        r"G:\test_data\new_data\new_data_from_whole_body_v2_6\annotations\person_keypoints_val2017.json"
    save_path_ = r"G:\test_data\new_data\new_data_from_whole_body_v2_6\annotations\person_keypoints_val2017_update.json"
    main(ignore_path_, json_path_, save_path_)