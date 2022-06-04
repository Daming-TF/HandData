import argparse
import os
from tqdm import tqdm

from json_tools import load_json_data, get_ids, write_json


def main(args):
    json_path = args.JsonPath
    direname, basename = os.path.split(json_path)
    save_path = os.path.join(direname, basename.split('.')[0]+'-update.json')
    images_dict, annotations_dict = load_json_data(json_path)
    image_ids = get_ids(images_dict)

    for image_id in tqdm(image_ids):
        annotations_info_list = annotations_dict[image_id]
        if len(annotations_info_list) == 2:
            continue
        for annotations_info in annotations_info_list:
            annotations_info['hand_type'] = 'right'

    write_json(images_dict, annotations_dict, save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--JsonPath", default=r'E:\v2_6\output\annotations\train2017\2022-5-31-9-53-55.json')
    args = parser.parse_args()
    main(args)
