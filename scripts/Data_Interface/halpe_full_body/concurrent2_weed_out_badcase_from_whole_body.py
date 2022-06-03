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


badcase_txt = r'E:\Data\landmarks\HFB\total_badcase_train.txt'
badcase_image_path = r'E:\Data\landmarks'
# image_path = r'E:\Data\landmarks\HFB\halpe_Full-Body_Human_Keypoints_and_HOL-Det_dataset\hico_20160224_det\images'
image_path = r'E:\Data\landmarks\HFB\halpe_Full-Body_Human_Keypoints_and_HOL-Det_dataset\hico_20160224_det\images'

json_path = r'E:\Data\landmarks\HFB\test\sort_halpe_train_v1.json'
# crop_json_path = r'E:\Data\landmarks\HFB\HFB\annotations\person_keypoints_train2017.json'
save_path = r'E:\Data\landmarks\HFB\halpe_Full-Body_Human_Keypoints_and_HOL-Det_dataset\badcase'
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


def run_process(badcase_image_dir_list, image_json_info, annotations_json_info, badcase_num, block_index, mode):
    print(f'This is {block_index} process')

    save_file = os.path.split(badcase_txt)[1]
    save_dir = os.path.join(save_path, 'blocks_' + str(block_index) + '_' + save_file)
    badcase_resave_dir = os.path.join(save_path, 'blocks_resave_' + str(block_index) + '_' + save_file)

    image_crop = ImageCrop(image_path, save_dir, badcase_resave_dir, image_json_info, annotations_json_info,
                           badcase_num, block_index, mode)

    # 遍历badcase
    for badcase_dir in badcase_image_dir_list:
        badcase_image_dir = os.path.join(badcase_image_path, badcase_dir)
        # badcase_image_dir = os.path.join(badcase_image_path, '.' + badcase_dir)
        image_crop.search(badcase_image_dir)

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
    print(f"loading the whole body json:{json_path}")
    with open(json_path, 'r') as whole_json:
        whole_json_data = json.load(whole_json)
        image_json_info = whole_json_data['images']
        annotations_json_info = whole_json_data['annotations']
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

        #进程并发
        print()
        # 父进程创建Queue队列，并传给各个子进程：
        # num_len = 20000
        # queue_info = Queue(num_len)
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
    mode = 'HFB-train'
    process_num = 12  # 进程数
    parallel_process_num = 4
    main()
