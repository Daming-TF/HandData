"""
V2-6数据清洗工具威力加强最终测试版
功能键：
    ’f‘：载入当前图片信息到json数据结构里面，并跳转到下一张图片
            当前图片还没标注时无法跳转到一下一张
    ’b‘：从json数据结构里头移除当前图片信息，并跳转到上一张图片
    ’k‘：把annotations[’hand_type]信息改为左手
    ’l‘：把annotations[’hand_type]信息改为右手
    ‘m’：记录需要打包给标注团队的数据信息
        格式如：{000014000001.jpg}**{mode}**{path}————path表示原图路径
    ‘p’：记录需要无效数据信息格式如上
"""
import argparse
import os
import json
import numpy as np
import cv2
import time
from copy import deepcopy
import shutil
from collections import defaultdict

from library.tools import draw_2d_points
from library.json_tools import make_json_head, convert_landmarks


def get_args():
    mode = 'train'
    parser = argparse.ArgumentParser()
    parser.description = 'Please enter two parameters a and b ...'

    # 需要标注的json文件
    # E:\v2_6\annotations\person_keypoints_{mode}2017.json
    parser.add_argument("--JsonPath",
                        default=fr"E:\v2_6\annotations\person_keypoints_{mode}2017.json",
                        help="this is parameter about the PATH of HandDet Model")

    # 存放上一次标记到的位置
    parser.add_argument("--LocatePath",
                        default=fr"E:\v2_6\output\location\{mode}2017\location.txt",
                        help="this is parameter about the PATH of Hand2D Model")

    # 存放标记好的json文件
    parser.add_argument("--SaveDir",
                        default=fr"E:\v2_6\output\annotations\{mode}2017",
                        help="this is parameter about the PATH of Hand2D Model")

    # 记录无效数据的数据路径
    parser.add_argument("--InvalidDir",
                        default=fr"E:\v2_6\invalid_data\{mode}2017\images",
                        help="this is parameter about the PATH of Hand2D Model")

    # 记录需要重新标注数据的图片路径
    parser.add_argument("--RemarkDir",
                        default=fr"E:\v2_6\remark_data\{mode}2017\images",
                        help="this is parameter about the PATH of Hand2D Model")
    args = parser.parse_args()

    os.makedirs(os.path.split(args.LocatePath)[0], exist_ok=True)
    os.makedirs(args.SaveDir, exist_ok=True)
    os.makedirs(args.InvalidDir, exist_ok=True)
    os.makedirs(args.RemarkDir, exist_ok=True)

    return args.JsonPath, args.LocatePath, args.SaveDir, args.InvalidDir, args.RemarkDir, mode


def get_location_info(txt_dir):
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
    json_path, locate_path, save_dir, invalid_dir, remark_dir, mode = get_args()
    invalid_txt_dir = os.path.join(os.path.split(invalid_dir)[0], "record.txt")
    remark_txt_dir = os.path.join(os.path.split(remark_dir)[0], "record.txt")

    if not os.path.exists(locate_path):
        start_index = 0
    else:
        record_info = get_location_info(locate_path)
        start_index = int(record_info[1])

    start_time, _ = get_time()
    index = start_index
    have_mark_count = 0
    skip_count = 0

    with open(json_path, 'r')as f:
        coco_data = json.load(f)
        images, annotations = convert_json(coco_data)
        assert (len(images) == len(annotations))

    info_num = len(images)

    if index == info_num:
        print("The all pic from this json file had been marked! Congratulation~")

    while 1:
        exit_flag = 0
        image_info = images[index]
        annotation_info = annotations[index]

        image_path = image_info['image_dir']
        file_name = image_info['file_name']
        width = image_info['width']
        height = image_info['height']

        if check_flag(annotation_info):
            print(f"index:{index}\tfile name:{file_name}\thand_type:'{annotation_info['hand_type']}'\t{image_path}")
        else:
            print(f"index:{index}\tfile name:{file_name}\tThis pic have not been marked")

        keypoints = np.array(annotation_info['keypoints']).reshape(21, 3)
        x, y, h, w = annotation_info['bbox']
        hand_type = annotation_info['hand_type']

        image = cv2.imread(image_path)
        img = draw_2d_points(keypoints, deepcopy(image))
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

        t_x, t_y = convert_landmarks((x, y, h, w), height, width)
        img = cv2.putText(img, hand_type, (t_x, t_y), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 125, 255), 2)

        canve = np.hstack([img, image])
        cv2.imshow('left_label_mark', canve)

        while 1:
            i = 0
            skip_flag = 0
            key = cv2.waitKey(0) & 0xFF

            if key == ord('f'):
                i += 1
                break
            elif key == ord('b'):
                if have_mark_count == 0:
                    print("This is index 0 pic")
                else:
                    i -= 1
                break
            elif key == ord('k'):
                annotation_info['hand_type'] = 'left'
                print(f"index:{index}\thand_type:'left'\t{image_path}")

            elif key == ord('l'):
                annotation_info['hand_type'] = 'right'
                print(f"index:{index}\thand_type:'right'\t{image_path}")

            elif key == ord('q'):
                exit_flag = 1
                break

            elif key == ord('m'):
                image_save_path = os.path.join(remark_dir, file_name)
                if not os.path.exists(image_save_path):
                    cv2.imwrite(image_save_path, canve)
                    with open(remark_txt_dir, 'a') as f:
                        f.write(file_name + '**' + mode + '**' + image_path + '\n')
                else:
                    print(f"({os.path.basename(image_path)})has beeen recored as >>Remark Data<<")
                break

            elif key == ord('p'):
                image_save_path = os.path.join(invalid_dir, file_name)
                if not os.path.exists(image_save_path):
                    cv2.imwrite(image_save_path, canve)
                    with open(invalid_txt_dir, 'a') as f:
                        f.write(file_name + '**' + mode + '**' + image_path + '\n')
                else:
                    print(f"({os.path.basename(image_path)})has beeen recored as >>Invalid Data<<")
                break

        annotation_info['mark_flag'] = 1

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
            time_info, time_id = get_time()

            assert (index - start_index - skip_count == len(json_head['images']))
            print(f"From //{start_time}// to //{time_info}// \t====> You mark {index-start_index-skip_count} pic")

            print("Writing the json file......")
            save_name = time_id + '.json'
            save_dir = os.path.join(save_dir, save_name)
            with open(save_dir, 'w')as sf:
                json.dump(json_head, sf)
            print(f"Succeed in writing the {save_dir}")
            with open(locate_path, 'w')as tf:
                tf.write(f"{time_info}\n{index}\n{json_path}\n{info_num}\n")
            print(f"Succeed in writing the {locate_path}")
            break


def convert_json(coco_data):
    annotations_dict = defaultdict(list)
    images_dict = {}

    for ann in coco_data['annotations']:
        annotations_dict[ann['image_id']].append(ann)

    for img in coco_data['images']:
        images_dict[img['id']] = img

    json_data = make_json_head()
    images_ids = list(images_dict.keys())
    for image_id in images_ids:
        image_info = images_dict[image_id]

        annotation_info_list = annotations_dict[image_id]
        for annotation_info in annotation_info_list:
            json_data["images"].append(image_info)
            json_data["annotations"].append(annotation_info)

    return json_data['images'], json_data['annotations']

if __name__ == '__main__':
    main()