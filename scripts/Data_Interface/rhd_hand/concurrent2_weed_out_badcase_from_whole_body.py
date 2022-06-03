import os
import pickle
import cv2
from tqdm import tqdm
from json_tools import crop_box
from vis_hand import coordinate_normalization
import numpy as np
from tools import draw_2d_points
from multiprocessing import Process
import sys
sys.path.append("..")
from weed_out_tool import ImageCrop


badcase_txt = r'F:\image\Rendered Handpose Dataset Dataset\RHD\RHD_published_v2\crop-image-badcase.txt'        # crop-image-badcase.txt
badcase_image_path = r'F:\image\Rendered Handpose Dataset Dataset'       # crop-image路径(该路径指向+crop-image-badcase.txt可以指向badcase)
image_path = r'F:\image\Rendered Handpose Dataset Dataset\RHD\RHD_published_v2'        # 与全图数据路径有密切关联

# json_path = r'F:\image\CMU\hand_labels\hand_labels'     # CMU每张数据原图对应一个json文件
save_path = r'F:\image\Rendered Handpose Dataset Dataset\RHD\RHD_published_v2\badcase'        # 数据保存路径

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
    annotations_json_info = []
    file_names_first_level = os.listdir(image_path)
    for file_name_first_level in file_names_first_level:
        if not os.path.isdir(os.path.join(image_path, file_name_first_level)) \
                or file_name_first_level not in ["evaluation", "training"]:
            continue
        pickle_dir = os.path.join(image_path, file_name_first_level, 'anno_%s.pickle' %file_name_first_level)
        with open(pickle_dir, 'rb')as fi:
            pickle_info = pickle.load(fi)
            # iterate samples of the set
            for sample_id, anno in tqdm(pickle_info.items()):
                img_dir = os.path.join(image_path, file_name_first_level, 'color', '%.5d.png' % sample_id)
                hands = list()
                if np.any(anno['uv_vis'][:21, 2]) == 1:
                    kp = coordinate_normalization(anno['uv_vis'][:21])
                    if not np.all(kp[2]) == 0:
                        hands.append(kp)
                if np.any(anno['uv_vis'][21:42, 2]) == 1:
                    kp = coordinate_normalization(anno['uv_vis'][21:42])
                    if not np.all(kp[2]) == 0:
                        hands.append(kp)

                if not os.path.exists(img_dir):
                    continue
                image_json_info.append(img_dir)
                annotations_json_info.append(hands)

    print(f'Successed to load all the image <==> There are {len(image_json_info)} pic date')
    block_len = len(image_json_info) // process_num
    demarcation = 0
    image_blocks, annotations_blocks = [], []
    for i in range(process_num):
        if not i == process_num - 1:
            image_block = image_json_info[demarcation: demarcation + block_len]
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
    process_list = [Process(target=run_process, args=(badcase_image_dir_list, image_blocks[i],
                                                      annotations_blocks[i], num, i, mode))for i in range(process_num)]

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
    mode = 'RHD'
    process_num = 12     # 进程数
    parallel_process_num = 4
    main()
