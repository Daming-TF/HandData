"""
功能：对比batch数据覆盖前后差别，用于检查标注团队标注质量的工具
"""
import cv2
import os
import numpy as np

from library.tools import draw_2d_points
from library.json_tools import load_json_data


def main():
    cv2.namedWindow("check", cv2.WINDOW_NORMAL)
    path = r'G:\test_data\hardcase_data\hardcase_from_whole_body_v2_6'
    new_json_path = fr'{path}\annotations\person_keypoints_train2017_update_with_beach_sample.json'
    old_json_path = fr'{path}\annotations\person_keypoints_train2017.json'
    match_txt_path = fr'{path}\annotations\train_batch_data\record.txt'
    data_dir = fr'{path}\images\train2017'

    new_images_dict, new_annotations_dict = load_json_data(new_json_path)
    old_images_dict, old_annotations_dict = load_json_data(old_json_path)

    image_id_list = load_txt_data(match_txt_path)

    sort_id = 0
    exit_flag = 0
    while 1:
        image_id = int(image_id_list[sort_id])

        new_image = get_image(new_images_dict, new_annotations_dict, image_id, data_dir)
        old_image = get_image(old_images_dict, old_annotations_dict, image_id, data_dir)

        canves = np.hstack([new_image, old_image])

        print(image_id)
        cv2.imshow("check", canves)
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

        if sort_id == len(image_id_list) or exit_flag == 1:
            print('Finished checking all the data')
            break


def get_ids(data_dict):
    return list(data_dict.keys())


def load_txt_data(match_txt_path):
    info_list = []
    with open(match_txt_path, 'r')as f:
        for line_info in f.readlines():
            line_info = line_info.strip('\n')
            info_list.append(line_info.split('**')[0])
    return info_list


def get_image(images_dict, annotations_dict, image_id, data_dir):
    image_info = images_dict[image_id]
    annotations_info_list = annotations_dict[image_id]

    image_id = image_info["id"]
    file_name = image_info["file_name"]
    image_path = os.path.join(data_dir, file_name)

    image = cv2.imread(image_path)

    for annotations_info in annotations_info_list:
        keypoints = annotations_info["keypoints"]
        annotation_id = annotations_info["image_id"]
        assert (image_id==annotation_id)
        if "hand_type" not in annotations_info.keys():
            print(f"{image_id}:[!] no hand_type")

        else:
            hand_type = annotations_info["hand_type"]
            bbox = annotations_info["bbox"]
            image = cv2.putText(image, hand_type, (int(bbox[0] + bbox[2] / 2), int(bbox[1] + bbox[3] - 20)),
                                cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 2)
        image = draw_2d_points(np.array(keypoints).reshape(21, 3), image)
    return image


if __name__ == "__main__":
    main()
