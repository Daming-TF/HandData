import copy
import json
import os
import shutil
import numpy as np
from tqdm import tqdm
import cv2
from json_tools import crop_box
from tools import draw_2d_points
from multiprocessing import Process
import _thread
import sys
sys.path.append("..")
from weed_out_tool import ImageCrop


badcase_txt = r'E:\Data\landmarks\handpose_x_gesture_v1\handpose_x_gesture_v1\total-badcase.txt'
badcase_image_path = r'E:\Data\landmarks'
image_path = r'E:\Data\landmarks\handpose_x_gesture_v1\handpose_x_gesture_v1'       # r'F:\image\coco\train'

json_path = r'E:\Data\landmarks\handpose_x_gesture_v1\handpose_x_gesture_v1'      # r'F:\image\coco\sort_coco_wholebody_train_v1.0.json'
save_path = r'E:\Data\landmarks\handpose_x_gesture_v1\handpose_x_gesture_v1\badcase'
# suspect_record_path = r'E:\Data\landmarks\HFB\halpe_Full-Body_Human_Keypoints_and_HOL-Det_dataset\suspect_image'

def check(landmarks, crop_landmarks, img):
    coco_kps = landmarks.copy()
    coco_kps[:, :2] = crop_landmarks[:, :2]
    coco_kps[:, 2] = landmarks[:, 2]
    kps_valid_bool = coco_kps[:, -1].astype(bool)
    coco_kps[~kps_valid_bool, :2] = 0
    img = draw_2d_points(coco_kps, img)
    cv2.imshow('show', img)
    cv2.waitKey(0)

NUM_JOINTS = 21
def convert_joints(hand_parts):
    joints = np.zeros((NUM_JOINTS, 3), dtype=float)
    for i in range(NUM_JOINTS):
        joints[i, 0] = hand_parts[str(i)]['x']
        joints[i, 1] = hand_parts[str(i)]['y']
        joints[i, 2] = hand_parts[str(i)]['score']
    return joints


def run_process(badcase_image_dir_list, image_json_info, annotations_json_info, badcase_num, block_index, mode):
    print(f'This is {block_index} process')

    save_file = os.path.split(badcase_txt)[1]
    save_dir = os.path.join(save_path, 'blocks_' + str(block_index) + '_' + save_file)
    badcase_resave_dir = os.path.join(save_path, 'blocks_resave_' + str(block_index) + '_' + save_file)

    image_crop = ImageCrop(image_path, save_dir, badcase_resave_dir, image_json_info, annotations_json_info,
                           badcase_num, block_index, mode)

    # 遍历badcase
    for badcase_dir in badcase_image_dir_list:
        badcase_image_dir = os.path.join(badcase_image_path,  badcase_dir)
        # image_crop.search(badcase_image_dir)

    print(f"This is {len(badcase_image_dir_list)} badcase image")
    print(f'find matched {image_crop.get_count}===> finish process')


def main():
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # 得到所有badcase的路径
    badcase_image_dir_list = list()
    with open(badcase_txt, 'r')as f_txt:
        badcase_info = f_txt.readlines()
    for i in range(len(badcase_info)):
        badcase_image_dir = badcase_info[i].split('\n')[0]
        badcase_image_dir_list.append(badcase_image_dir)
        num = len(badcase_image_dir_list)

    # 得到全身的json数据
    image_json_info, annotations_json_info = [], []
    print(f"loading the whole body json:{json_path}")
    sub_folders = sorted(os.path.join(json_path, file) for file in os.listdir(json_path)
                         if not file.endswith('.txt') and file != 'badcase')

    print('loading the json date......')
    json_dirs = []
    for sub_folder in sub_folders:
        json_dirs.extend(sorted(os.path.join(sub_folder, file) for file in os.listdir(sub_folder)
                                 if file.endswith('.json')))

    for i, json_dir in enumerate(json_dirs):
        img_dir = json_dir.replace('.json', '.jpg')

        anno_infos = json.load(open(json_dir, "r"))
        if 'error_msg' not in anno_infos:
            num_hands = len(anno_infos["hand_info"])
        else:
            continue

        if num_hands != 1:
            continue

        anno_info = anno_infos["hand_info"][0]
        hand_parts = anno_info["hand_parts"]
        joints = convert_joints(hand_parts)

        image_json_info.append(img_dir)
        annotations_json_info.append(joints)

    print(f'annotations length is {len(annotations_json_info)} pic date')

    # 划分每个进程的数据
    block_len = len(image_json_info) // process_num
    demarcation = 0
    image_blocks, annotations_blocks = [], []
    for i in range(process_num):
        if not i == process_num-1:
            image_block = image_json_info[demarcation: demarcation+block_len]
            image_blocks.append(image_block)
            annotations_block = annotations_json_info[demarcation: demarcation + block_len]
            annotations_blocks.append(annotations_block)
            demarcation = demarcation + block_len

        else:
            image_block = image_json_info[demarcation:]
            image_blocks.append(image_block)
            annotations_block = annotations_json_info[demarcation:]
            annotations_blocks.append(annotations_block)

    del image_json_info, annotations_json_info

    # 进程并发
    print()
    # 父进程创建Queue队列，并传给各个子进程：
    process_list = [Process(target=run_process,
                            args=(badcase_image_dir_list, image_blocks[i], annotations_blocks[i], num, i, mode))
                    for i in range(process_num)]

    i = 0
    while 1:
        for j in range(parallel_process_num):
            process_list[i + j].start()
        for j in range(parallel_process_num):
            process_list[i + j].join()

        print(f"index of process is{i}")
        if (i + parallel_process_num) == process_num:
            print("SUCCESSED!!")
            break
        i += parallel_process_num




if __name__ == '__main__':
    mode = 'HXG'
    process_num = 12  # 进程数
    parallel_process_num = 1
    main()
