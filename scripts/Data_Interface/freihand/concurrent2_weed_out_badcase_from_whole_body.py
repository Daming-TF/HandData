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
from fh_utils import db_size, load_db_annotation, read_img_cv, projectPoints


badcase_txt = r'E:\Data\landmarks\FH\total-badcase.txt'
badcase_image_path = r'E:\Data\landmarks'
image_path = r'E:\Data\landmarks\FreiHAND_pub_v2'       # r'F:\image\coco\train'

json_path = r'E:\Data\landmarks\FreiHAND_pub_v2'      # r'F:\image\coco\sort_coco_wholebody_train_v1.0.json'
save_path = r'E:\Data\landmarks\FreiHAND_pub_v2\badcase'
# suspect_record_path = r'E:\Data\landmarks\HFB\halpe_Full-Body_Human_Keypoints_and_HOL-Det_dataset\suspect_image'

VERIONS = ['gs', 'hom', 'sample', 'auto']

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
    print(f"loading the whole body json:{json_path}")
    num_freqs = db_size('training')     # 返回单个版本的数据量
    num_total_imgs = 4 * num_freqs      # 所有数据数量
    indexs = np.arange(num_total_imgs)  # index是对应每张图片的序号的列表

    # load annotations
    # BASE_PATH = r"D:\Data\landmarks\FreiHAND_pub_v2\training"
    db_data_anno = list(load_db_annotation(image_path, 'training'))

    image_json_info, annotations_json_info = [], []
    for i in tqdm(range(indexs.shape[0])):
        index = indexs[i]
        version = VERIONS[index // num_freqs]  # VERIONS = ['gs', 'hom', 'sample', 'auto']
        # load image and mask
        img_path = read_img_cv(index, image_path, 'training', version)
        image_json_info.append(img_path)

        # annotation for this frame
        K, mano, xyz = db_data_anno[index % num_freqs]
        K, mano, xyz = [np.array(x) for x in [K, mano, xyz]]
        joints = projectPoints(xyz, K)
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
    mode = 'FH'
    process_num = 12  # 进程数
    parallel_process_num = 3
    main()
