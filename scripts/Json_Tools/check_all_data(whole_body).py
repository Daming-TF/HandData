import json
import os
import cv2
import numpy as np
from library.tools import draw_2d_points
import copy
from json_tools import crop_box

def get_data_name_index(image_id):
    if 300_000 <= image_id < 400_000:
        return 0
    elif 400_000 <= image_id < 500_000:
        return 1
    elif 500_000 <= image_id < 600_000:
        return 2
    elif 600_000 <= image_id < 800_000:
        return 3
    elif 800_000 <= image_id < 900_000:
        return 4
    elif 900_000 <= image_id < 1_000_000:
        return 5
    elif 1_000_000 <= image_id < 1_100_000:
        return 6
    elif 1_100_000 <= image_id < 1_200_000:
        return 7
    elif 1_200_000 <= image_id < 1_300_000:
        return 8
    elif 1_300_000 <= image_id < 1_400_000:
        return 9
    elif 1_400_000 <= image_id < 1_500_000:
        return 10
    elif 1_500_000 <= image_id < 1_600_000:
        return 11

def main():
    mode_list = ['train', 'val', 'test']
    # Ho3d: G:\imgdate2\HO3D_v3\HO3D_from_whole_body\annotations\person_keypoints_val2017-update-for-invaliddata.json
    # E:\left_hand_label_data\annotations\ho3d_update\person_keypoints_val2017.json
    # E:\left_hand_label_data\annotations\person_keypoints_val2017.json
    newjson_dir = r'E:\left_hand_label_data\annotations\person_keypoints_val2017.json'

    # G:\imgdate2\HO3D_v3\HO3D_from_whole_body\images\val2017
    data_path_list = [
        r'E:\Data\landmarks\YouTube3D\YouTube3D-wholebody',
        r'E:\Data\landmarks\HFB\HFB_from_whole_body',
        r'E:\Data\landmarks\handpose_x_gesture_v1\HXG_from_whole_body',
        r'E:\Data\landmarks\FH\FH_from_whole_body',
        r'F:\image\CMU\hand_labels\hand_labels_from_whole_body',
        r'F:\image\CMU\hand143_panopticdb\hand143_pannopticdb_from_whole_body',
        r'F:\image\CMU\hand_labels_synth\hand_labels_synth_from_whole_body',
        r'F:\image\Rendered Handpose Dataset Dataset\RHD\RHD_from_whole_body',
        r'F:\image\COCO_whole_body\coco_from_whole_body',
        r'G:\imgdate2\HO3D_v3\HO3D_from_whole_body',
        r'G:\test_data\new_data\new_data_from_whole_body',
        r'G:\test_data\hardcase_data\hardcase_from_whole_body',
                 ]

    save_record_path = r'E:\left_hand_label_data\record'
    save_invalid_path = r'E:\left_hand_label_data\invalid'
    badcase_txt_name = r'badcase.txt'
    mode = ''

    for mode_name in mode_list:
        if mode_name in newjson_dir:
            mode = mode_name

    for path in [save_invalid_path, save_record_path]:
        if not os.path.exists(path):
            os.mkdir(path)
    #
    cv2.namedWindow('show', cv2.WINDOW_NORMAL)
    with open(newjson_dir, 'r') as f:
        json_labels = json.load(f)
        images = json_labels['images']
        annotations = json_labels['annotations']

        data_num = len(images)

        exit_flag = 0
        i = 0
        while 1:
            image_info = images[i]
            annotations_info = annotations[i]
            file_name = image_info['file_name']
            image_id = annotations_info['image_id']
            box = annotations_info['bbox']

            x, y, w, h = box

            index = get_data_name_index(int(image_id))
            data_path = data_path_list[index]

            image_dir = os.path.join(data_path, "images", f"{mode}2017", file_name)
            keypoints = annotations_info['keypoints']


            image = cv2.imread(image_dir)
            image1 = copy.deepcopy(image)
            image1 = draw_2d_points(np.array(keypoints).reshape([21, 3]), image1)
            if "hand_type" in annotations_info.keys():
                hand_type = annotations_info['hand_type']
                image1 = cv2.putText(image1, f'{hand_type}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
            image1 = cv2.rectangle(image1, (x, y), (x + w, y + h), (0, 0, 255), 2)

            print(f'image name:{image_dir}\t index:{i}\tremain:{data_num - i - 1}')

            canvas = np.hstack([image1, image])
            cv2.imshow('show', canvas)

            while 1:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('f'):
                    i = i + 1
                    break
                elif key == ord('b'):
                    i = i - 1
                    break
                elif key == ord('m'):
                    # 检查图片是否已经标记
                    txt_path = os.path.join(save_record_path, badcase_txt_name)
                    img_examine_path = os.path.join(save_record_path, os.path.basename(image_dir))
                    if not os.path.exists(img_examine_path):
                        cv2.imwrite(img_examine_path, image1)
                        with open(txt_path, 'a') as f:
                            f.write(image_dir+'\n')

                    else:
                        print(f"({os.path.basename(image_dir)})has beeen recored")

                    i = i + 1
                    break

                # 无效样本
                elif key == ord('p'):
                    # 检查图片是否已经标记
                    txt_path = os.path.join(save_invalid_path, badcase_txt_name)
                    img_examine_path = os.path.join(save_invalid_path, os.path.basename(image_dir))
                    if not os.path.exists(img_examine_path):
                        cv2.imwrite(img_examine_path, image1)
                        with open(txt_path, 'a') as f:
                            f.write(image_dir+'\n')

                    else:
                        print(f"({os.path.basename(image_dir)})has beeen recored")

                    i = i + 1
                    break

                if key == ord('a'):
                    i = i - 100
                    break
                if key == ord('z'):
                    i = i + 100
                    break

                if key == ord('s'):
                    i = i - 1000
                    break
                if key == ord('x'):
                    i = i + 1000
                    break

                if key == ord('d'):
                    i = i - 10000
                    break
                if key == ord('c'):
                    i = i + 10000
                    break

                elif key == ord('q'):
                    exit_flag = 1
                    break

            if i == len(images) or exit_flag == 1:
                print('Finished checking all the data')
                break

if __name__ == '__main__':
    main()