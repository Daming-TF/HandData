import argparse
import os
from tqdm import tqdm

from json_tools import load_json_data, get_ids, write_json


def main(args):
    json_path = args.JsonPath
    images_dict, annotations_dict = load_json_data(json_path)
    image_ids = get_ids(images_dict)

    record_list = []
    for image_id in tqdm(image_ids):
        if 1_200_000 > image_id >= 1_100_000:
            annotations_info_list = annotations_dict[image_id]
            if len(annotations_info_list) != 2:
                continue
            if annotations_info_list[0]['hand_type'] == annotations_info_list[1]['hand_type']:
                record_list.append(image_id)

    print(f"len:{len(record_list)}")
    for i in range(len(record_list)):
        print(str(record_list[i])+'\t', end='')
        if i % 12 == 0:
            print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--JsonPath", default=r'E:\v2_6\annotations\person_keypoints_train2017.json')
    args = parser.parse_args()
    main(args)
