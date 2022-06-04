# 1410547, 1412471, 1412476, 1412539, 1414189
import json

from library.json_tools import load_json_data, write_json

del_list = [1410547, 1412471, 1412476, 1412539, 1414189]
json_path = r'E:\test_data\test_data_from_whole_body\annotations\average_pseudo_labels_update-coco_id.json'

def main():
    images_dict, annotations_dict = load_json_data(json_path)
    for del_index in del_list:
        images_dict.pop(del_index)
        annotations_dict.pop(del_index)

    write_json(images_dict, annotations_dict, json_path)


if __name__ == "__main__":
    main()
