import argparse
import os
import json
import numpy as np
from tools import draw_2d_points
from library.json_tools import make_json_head
import cv2
import time
from copy import deepcopy
import shutil


def get_args():
    mode = 'test'
    parser = argparse.ArgumentParser()
    parser.description = 'Please enter two parameters a and b ...'

    # 需要标注的json文件
    parser.add_argument("--JsonDir",
                        default=fr"E:\left_hand_label_data\annotations\v2_6\test_data_add_hand\mediapipe-detecter\badcase_test.json",
                        help="this is parameter about the PATH of HandDet Model")

    # 存放上一次标记到的位置
    parser.add_argument("--TxtDir",
                        default=fr"E:\left_hand_label_data\annotations\v2_6\test_data_add_hand\manual_tagging\record.txt",
                        help="this is parameter about the PATH of Hand2D Model")

    # 存放标记好的json文件
    parser.add_argument("--SavePath",
                        default=fr"E:\left_hand_label_data\annotations\v2_6\test_data_add_hand\manual_tagging",
                        help="this is parameter about the PATH of Hand2D Model")

    args = parser.parse_args()
    return args


def get_txt_info(txt_dir):
    txt_path, txt_name = os.path.split(txt_dir)
    backup_dir = os.path.join(txt_path, txt_name.split('.')[0]+"_backup.txt")
    shutil.copyfile(txt_dir, backup_dir)
    return_list = list()
    with open(txt_dir, 'r') as f_txt:
        info_list = f_txt.readlines()
        if len(info_list) == 0:
            return [0, 0]
    for i in range(len(info_list)):
        info = info_list[i].split('\n')[0]
        return_list.append(info)

    return return_list


def check_flag(annotation_info):
    return 1 if 'mark_flag' in annotation_info.keys() else 0


def get_time():
    time_info = time.localtime(time.time())
    tm_year, tm_mon, tm_day, tm_hour, tm_min, tm_sec = time_info[0: 6]
    time_info = f"{tm_year}.{tm_mon}.{tm_day}\t{tm_hour}:{tm_min}:{tm_sec}"
    file_name = f"{tm_year}-{tm_mon}-{tm_day}-{tm_hour}-{tm_min}-{tm_sec}"
    return time_info, file_name


def main():
    cv2.namedWindow('left_label_mark', cv2.WINDOW_NORMAL)
    json_head = make_json_head()
    args = get_args()
    json_dir, txt_dir, save_path = args.JsonDir, args.TxtDir, args.SavePath
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    if not os.path.exists(txt_dir):
        start_index = 0
    else:
        record_info = get_txt_info(txt_dir)
        start_index = int(record_info[1])

    start_time, _ = get_time()
    index = start_index
    have_mark_count = 0
    skip_count = 0

    with open(json_dir, 'r')as f:
        json_data = json.load(f)
        images = json_data['images']
        annotations = json_data['annotations']
        assert (len(images) == len(annotations))

    info_num = len(images)

    if index == info_num:
        print("The all pic from this json file had been marked! Congratulation~")

    while 1:
        exit_flag = 0
        image_info = images[index]
        annotation_info = annotations[index]
        image_dir = image_info['image_dir']

        if check_flag(annotation_info):
            print(f"index:{index}\thand_type:'{annotation_info['hand_type']}'\t{image_dir}")
        else:
            print(f"index:{index}\tThis pic have not been marked")
        keypoints = np.array(annotation_info['keypoints']).reshape(21, 3)
        image = cv2.imread(image_dir)
        img = draw_2d_points(keypoints, deepcopy(image))
        canve = np.hstack([img, image])
        cv2.imshow('left_label_mark', canve)

        while 1:
            i = 0
            skip_flag = 0
            key = cv2.waitKey(0) & 0xFF

            if key == ord('f'):
                if not check_flag(annotation_info):     # 如果当前图片还没有标记无法跳到下一张
                    print("This pic have not been marked")
                else:
                    i += 1
                break
            elif key == ord('b'):
                if skip_count != 0:
                    skip_count -= 1
                i -= 1
                break
            elif key == ord('k'):
                annotation_info['hand_type'] = 'left'
                annotation_info['mark_flag'] = 1
                print(f"index:{index}\thand_type:'left'\t{image_dir}")

            elif key == ord('l'):
                annotation_info['hand_type'] = 'right'
                annotation_info['mark_flag'] = 1
                print(f"index:{index}\thand_type:'right'\t{image_dir}")

            elif key == ord('s'):
                i += 1
                skip_count += 1
                skip_flag = 1
                break

            elif key == ord('q'):
                exit_flag = 1
                break

        index += i
        have_mark_count += i
        if i < 0:
            del json_head['images'][have_mark_count:]
            del json_head['annotations'][have_mark_count:]
        elif i > 0 and skip_flag == 0:
            json_head['images'].append(image_info)
            json_head['annotations'].append(annotation_info)

        if index == info_num or exit_flag:
            print("ready to exit......")
            time_info, file_name = get_time()

            assert (index - start_index - skip_count == len(json_head['images']))
            print(f"From //{start_time}// to //{time_info}// \t====> You mark {index-start_index-skip_count} pic")

            print("Writing the json file......")
            save_name = file_name + '.json'
            save_dir = os.path.join(save_path, save_name)
            with open(save_dir, 'w')as sf:
                json.dump(json_head, sf)
            print(f"Succeed in writing the {save_dir}")
            with open(txt_dir, 'w')as tf:
                tf.write(f"{time_info}\n{index}\n{json_dir}\n{info_num}\n")
            print(f"Succeed in writing the {txt_dir}")
            break


if __name__ == '__main__':
    main()