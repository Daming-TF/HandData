"""
功能： 标记脸的左眼，右眼，鼻子，嘴巴4个点的可见性

    修改可见性的功能键介绍：
    ————按下功能键‘1’~‘4’则对应位置关键点可见性取反；  功能键‘1’~‘4’分别表示左眼，右眼，鼻子，嘴巴四个位置的可见性
    ————按下功能键‘r’表示当前帧可见性默认与上一帧一致；
    ————按下功能键‘q’表示当前帧所有关键点可见性置为1；
    ————按下功能键‘w’表示当前帧所有关键点可见性置为0；

    翻页功能键介绍：
    ————按下功能键‘k’表示当前帧已完成，把图片序号和可见性写入txt文件中，并显示下一帧
    ————按下功能键‘l’表示上一帧标记有问题，跳转到上一帧，并删除txt中记录的当前帧信息
"""

import cv2
import os
import argparse
import shutil
import numpy as np
import time
from copy import deepcopy


def main(args):
    cv2.namedWindow('check', cv2.WINDOW_NORMAL)
    image_names = os.listdir(args.image_dir)
    out_path = os.path.join(args.image_dir, "output")
    os.makedirs(out_path, exist_ok=True)
    record_path = os.path.join(out_path, args.record_info_txt)  # 记录上一次标注信息的txt文件路径
    output_path = os.path.join(out_path, os.path.basename(args.image_dir)+'.txt')  # 记录可见性编码的txt文件路径
    print(output_path)

    start_index = int(get_location_info(record_path)) if os.path.exists(record_path) else 0
    index = start_index
    last_info = None

    while 1:
        image_name = image_names[index]
        image_path = os.path.join(args.image_dir, image_name)
        image = cv2.imread(image_path)

        print(f"index:{index}")
        cv2.imshow('check', image)

        visibility_info = last_info if last_info is not None else np.array(np.ones(4), dtype=int)
        exit_flag = 0
        while 1:
            debug_img = draw_visibility(deepcopy(image), visibility_info)
            cv2.imshow('check', debug_img)
            key = cv2.waitKey(0) & 0xFF
            if key == ord('1'):
                visibility_info[0] = ~visibility_info[0]+2
            elif key == ord('2'):
                visibility_info[1] = ~visibility_info[1]+2
            elif key == ord('3'):
                visibility_info[2] = ~visibility_info[2]+2
            elif key == ord('4'):
                visibility_info[3] = ~visibility_info[3]+2

            elif key == ord('r'):
                visibility_info = last_info

            elif key == ord('q'):
                visibility_info = np.array(np.zeros(4), dtype=int)
            elif key == ord('w'):
                visibility_info = np.array(np.ones(4), dtype=int)


            if key == ord('k'):
                write_info(output_path, index, visibility_info)
                last_info = visibility_info
                index += 1
                break
            if key == ord('l'):
                del_info(output_path, index)
                index -= 1
                break
            elif key == ord('z'):
                exit_flag = 1
                break

        # print(f"index:{index}\timage_num:{len(image_names)}")
        if exit_flag or index >= len(image_names)-1:
            print("ready to exit......")
            time_info = get_time()
            with open(record_path, 'w') as tf:
                tf.write(f"{time_info}\n{str(index).zfill(5)}\n")
            print(f"Succeed in writing the {record_path}")
            break
    return 0


def get_location_info(txt_dir):
    txt_path, txt_name = os.path.split(txt_dir)
    backup_dir = os.path.join(txt_path, txt_name.split('.')[0]+"_backup.txt")
    shutil.copyfile(txt_dir, backup_dir)
    return_list = list()
    with open(txt_dir, 'r') as f_txt:
        info_list = f_txt.readlines()
        if len(info_list) == 0:
            return 0
    for i in range(len(info_list)):
        info = info_list[i].split('\n')[0]
        return_list.append(info)
    assert len(return_list) == 2

    return return_list[1]


def draw_visibility(image, info_numpy):
    for i in range(info_numpy.shape[0]):
        cv2.putText(image, str(int(info_numpy[i])), (50*(i+1), 50), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 125, 255), 2)
    return image


def get_time():
    time_info = time.localtime(time.time())
    tm_year, tm_mon, tm_day, tm_hour, tm_min, tm_sec = time_info[0: 6]
    time_info = f"{tm_year}.{tm_mon}.{tm_day}\t{tm_hour}:{tm_min}:{tm_sec}"
    # file_name = f"{tm_year}-{tm_mon}-{tm_day}-{tm_hour}-{tm_min}-{tm_sec}"
    return time_info


def write_info(output_path, index, visibility_info):
    with open(output_path, 'a')as f:
        f.write(f"{str(index).zfill(5)}\t")
        for i in range(visibility_info.shape[0]):
            f.write(f"{visibility_info[i]}")
        f.write("\n")
    print(f'index:{index}\tinfo:{list(visibility_info)}')


def del_info(output_path, index):
    new_info = []
    with open(output_path, 'r') as rf:
        info_list = rf.readlines()

    for i in range(len(info_list)):
        info = info_list[i].split('\n')[0]
        new_info.append(info)
    # print(int(new_info[-1].split('\t')[0]))
    # print(index)
    assert int(new_info[-1].split('\t')[0]) == index-1
    vis = new_info[-1].split('\t')[1]
    print(f"del index:{index - 1}\t\nnow index:{index - 1}\tinfo:{vis}")
    del new_info[-1]

    with open(output_path, 'w') as wf:
        for info in new_info:
            wf.write(f'{info}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", default=r'E:\faqiang\src\xulie_facial')
    parser.add_argument("--record_info_txt", default=r'record.txt')       # visibility
    # parser.add_argument("--visibility_info_txt", default=r'output.txt')
    args = parser.parse_args()

    main(args)
