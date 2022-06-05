import cv2
import json
import os
from collections import defaultdict
import numpy as np

from library.tools import draw_2d_points
from library.json_tools import convert_landmarks

debug = 1

def main():
    cv2.namedWindow("check", cv2.WINDOW_NORMAL)
    json_path = r'E:\v2_6\output\annotations\train2017\2022-5-30-21-59-13-update.json'
    data_dir = r''  # G:\test_data\new_data\new_data_from_whole_body_v2_6\images
    images_dict, annotations_dict = load_json_data(json_path)
    ids = get_ids(images_dict)

    del_list = []
    exit_flag = 0
    sort_id = 0

    while 1:
        image_id = ids[sort_id]
        print(f"image_id:{image_id}\tremain:{len(ids)-sort_id}")

        # if not image_id == 1400025:
        #     sort_id += 1
        #     continue

        flag = 1
        image_info = images_dict[image_id]
        annotations_info_list = annotations_dict[image_id]

        image_id = image_info['id']
        file_name = image_info["file_name"]
        image_path = os.path.join(data_dir, file_name)
        if not os.path.exists(image_path):
            image_path = image_info["image_dir"]
        image = cv2.imread(image_path)
        i_h, i_w = image.shape[:2]
        # print(image_id)

        for annotations_info in annotations_info_list:
            keypoints = annotations_info["keypoints"]
            annotation_id = annotations_info["image_id"]
            hand_type = annotations_info["hand_type"] if 'hand_type' in annotations_info.keys() else None
            x, y, h, w = annotations_info["bbox"]

            assert (image_id == annotation_id)

            if hand_type is None:
                print(f"{image_id}:[!] no hand_type")
                del_list.append(image_id)
                flag = 0
            else:
                hand_type = annotations_info["hand_type"]
                txt = 'r' if hand_type == 'right' else 'l'
                # int(bbox[0] + bbox[2] / 2), int(bbox[1] + bbox[3] - 20)
                t_x, t_y = convert_landmarks((x, y, h, w), i_h, i_w)
                image = cv2.putText(image, txt, (t_x, t_y),
                                    cv2.FONT_HERSHEY_COMPLEX, 2, (0, 125, 255), 2)
            image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            image = draw_2d_points(np.array(keypoints).reshape(21, 3), image)


        cv2.imshow("check", image)
        if debug == 0:
            cv2.waitKey(1) & 0xFF
            sort_id += 1

        while 1 and debug:
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