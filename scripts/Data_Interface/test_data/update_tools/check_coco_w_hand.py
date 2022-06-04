import cv2
import json
import os
from collections import defaultdict
import numpy as np
from library.tools import draw_2d_points


def main():
    cv2.namedWindow("check", cv2.WINDOW_NORMAL)
    json_path = r'G:\test_data\new_data\new_data_from_whole_body\match-v2_4-v2_5\v2_4_person_keypoints_test2017-update.json'
    data_dir = r'G:\test_data\new_data\crop_images'
    images_dict, annotations_dict = load_json_data(json_path)
    ids = get_ids(images_dict)

    del_list = []
    for image_id in ids:
        print(image_id)
        flag = 1
        image_info = images_dict[image_id]
        annotations_info_list = annotations_dict[image_id]

        file_name = image_info["file_name"]
        image_path = os.path.join(data_dir, file_name)
        image = cv2.imread(image_path)
        # print(image_id)

        for annotations_info in annotations_info_list:
            keypoints = annotations_info["keypoints"]
            if "hand_type" not in annotations_info.keys():
                print(f"{image_id}:[!] no hand_type")
                del_list.append(image_id)
                flag = 0
            else:
                hand_type = annotations_info["hand_type"]
                bbox = annotations_info["bbox"]
                image = cv2.putText(image, hand_type, (int(bbox[0] + bbox[2] / 2), int(bbox[1] + bbox[3] - 20)),
                                    cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)
            image = draw_2d_points(np.array(keypoints).reshape(21, 3), image)

        cv2.imshow("check", image)
        cv2.waitKey(1)

    print(del_list)


def get_ids(data_dict):
    return list(data_dict.keys())


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


if __name__ == "__main__":
    main()
