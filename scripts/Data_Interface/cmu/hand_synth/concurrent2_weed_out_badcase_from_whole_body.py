import os
import cv2
from multiprocessing import Process

import sys
sys.path.append("../../")
from library.weed_out_tool import ImageCrop
from library.json_tools import crop_box
from library.tools import draw_2d_points

badcase_txt = r'F:\image\CMU\hand_labels_synth\hand_labels_synth\crop-image-badcase.txt'        # crop-image-badcase.txt
badcase_image_path = r'F:\image\CMU'       # crop-image路径(该路径指向+crop-image-badcase.txt可以指向badcase)
image_path = r'F:\image\CMU\hand_labels_synth\hand_labels_synth'        # 与全图数据路径有密切关联

json_path = r'F:\image\CMU\hand_labels_synth\hand_labels_synth'     # CMU每张数据原图对应一个json文件
save_path = r'F:\image\CMU\hand_labels_synth\hand_labels_synth\badcase'        # 数据保存路径


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

    # 得到全身的数据
    print(f"loading the whole body data.........")
    image_json_info = []
    file_names_first_level = os.listdir(image_path)
    for file_name_first_level in file_names_first_level:
        for keystring in ['train', 'val', 'test']:
            if file_name_first_level.find(keystring) > 0:
                file_path = os.path.join(image_path, file_name_first_level)
                file_names_second_level = os.listdir(file_path)     # 图片以及json的文件的文件名
                for file_name_second_level in file_names_second_level:
                    if file_name_second_level.endswith('.jpg'):
                        img_dir = os.path.join(file_path, file_name_second_level)
                        image_json_info.append(img_dir)
    print(f'Successed to load all the image <==> There are {len(image_json_info)} pic date')

    block_len = len(image_json_info) // process_num
    demarcation = 0
    image_blocks, annotations_blocks = [], []
    for i in range(process_num):
        if not i == process_num - 1:
            image_block = image_json_info[demarcation: demarcation + block_len]
            image_blocks.append(image_block)
            demarcation = demarcation + block_len

        else:
            image_block = image_json_info[demarcation:]
            image_blocks.append(image_block)

    # 进程并发
    process_list = [Process(target=run_process, args=(badcase_image_dir_list,
                                                      image_blocks[i], None, num, i, mode))
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
    mode = 'CMU-synth'
    process_num = 12  # 进程数
    parallel_process_num = 4
    main()
