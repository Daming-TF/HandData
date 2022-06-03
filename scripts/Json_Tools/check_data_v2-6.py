import cv2
import json
import os
from collections import defaultdict
import numpy as np
from library.models.tools import draw_2d_points


def main():
    cv2.namedWindow("check", cv2.WINDOW_NORMAL)
    json_path = r'F:\image\CMU\hand_labels\hand_labels_from_whole_body_v2_6\annotations\annotations_update_with_beach_sample.json'
    data_dir = r'F:\image\CMU\hand_labels\hand_labels_from_whole_body_v2_6\images\val2017'
    images_dict, annotations_dict = load_json_data(json_path)
    ids = get_ids(images_dict)

    sort_id = 0
    exit_flag = 0
    while 1:
        image_id = ids[sort_id]
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
                flag = 0
            else:
                hand_type = annotations_info["hand_type"]
                bbox = annotations_info["bbox"]
                image = cv2.putText(image, hand_type, (int(bbox[0] + bbox[2] / 2), int(bbox[1] + bbox[3] - 20)),
                                    cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 2)
            image = draw_2d_points(np.array(keypoints).reshape(21, 3), image)

        print(image_id)
        cv2.imshow("check", image)
        while 1:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('a'):
                sort_id += 1
                break
            elif key == ord('z'):
                sort_id -= 1
                break

            if key == ord('s'):
                sort_id += 100
                break
            if key == ord('x'):
                sort_id -= 100
                break

            if key == ord('d'):
                sort_id += 1000
                break
            if key == ord('c'):
                sort_id -= 1000
                break

            if key == ord('f'):
                sort_id += 10000
                break
            if key == ord('v'):
                sort_id -= 10000
                break

            elif key == ord('q'):
                exit_flag = 1
                break

        if sort_id == len(ids) or exit_flag == 1:
            print('Finished checking all the data')
            break


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