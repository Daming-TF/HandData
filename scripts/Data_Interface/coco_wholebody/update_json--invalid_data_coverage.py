"""
    根据无效数据的json文件，对我们的coco-json文件删除匹配的图片信息
"""
import json
from tqdm import tqdm
import os
import argparse
import numpy as np
from convert_tools import convert_coco_format_from_wholebody

mode = 'train'

def get_pic_num(data_path):
    count = 0
    file_names = os.listdir(data_path)
    for file_name in file_names:
        path = os.path.join(data_path, file_name)
        if os.path.isdir(path):
            img_names = os.listdir(path)
            for img_name in img_names:
                if img_name.endswith('.jpg'):
                    count += 1
    return count




def get_file_list(file_path):
    dir_list = os.listdir(file_path)
    if not dir_list:
        return
    else:
        # 注意，这里使用lambda表达式，将文件按照最后修改时间顺序升序排列
        # os.path.getmtime() 函数是获取文件最后修改时间
        # os.path.getctime() 函数是获取文件最后创建时间
        dir_list = sorted(dir_list, key=lambda x: os.path.getmtime(os.path.join(file_path, x)))
        # print(dir_list)
        return dir_list

def set_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--TagJsonPath", help="无效数据的json文件路径",
                        default=r"E:\数据标记反馈\coco(6137)\无效样本")
    parser.add_argument("--DataPath", help="匹配覆盖对应路径图片的信息，所需要的原图数据路径",
                        default=r"F:\image\coco")
    parser.add_argument("--JsonDir", help="数据集json文件路径——cocoJson文件路径",
                        default=rf"F:\image\COCO_whole_body\coco_from_whole_body\annotations\person_keypoints_{mode}2017-update-for-batchdata.json")
    parser.add_argument("--JsonSaveDir", help="更新后文件写入路径",
                        default=rf"F:\image\COCO_whole_body\coco_from_whole_body\annotations\person_keypoints_{mode}2017-update-for-invaliddata.json")       # person_keypoints_{mode}-update.json
    args = parser.parse_args()
    return args

def get_keypoints(label_feature):
    handlandmarks_list = []
    hand1 = np.zeros((21, 3))
    hand2 = np.zeros((21, 3))
    keys = label_feature[0].keys()
    for key in keys:
        hand_info = key.split('-')
        hand_index = int(hand_info[0])
        landmark_index = int(hand_info[1])
        if hand_index == 0:
            hand1[landmark_index, :2] = np.array(label_feature[0][key][:2])
        elif hand_index == 1:
            hand2[landmark_index, :2] = np.array(label_feature[0][key][:2])
    if not np.all(hand1 == 0):
        hand1[:, 2] = 2
        handlandmarks_list.append(hand1)
    if not np.all(hand2 == 0):
        hand2[:, 2] = 2
        handlandmarks_list.append(hand2)
    return handlandmarks_list

def main():
    args = set_parser()
    tag_json_path = args.TagJsonPath        # 存放重标打回数据的路径
    json_dir = args.JsonDir         # 我们的coco-json文件路径
    data_path = args.DataPath       # 原图数据的原始路径
    json_save_dir = args.JsonSaveDir        # 更新后json保存路径

    # 读取原本json信息
    print(f'loading the file >> {json_dir}')
    with open(json_dir, 'r', encoding='UTF-8') as f:
        json_data = json.load(f)

    # 按时间序列返回json文件
    json_files = get_file_list(tag_json_path)
    print(f">> {len(json_files)} is invalid <<")
    for json_file in json_files:
        # 每次覆盖完，更新json信息
        images_list = json_data['images']

        tag_json_dir = os.path.join(tag_json_path, json_file)
        with open(tag_json_dir, 'r', encoding='UTF-8')as f:
            tag_json_date = json.load(f)
        print(f'There is >{ len(tag_json_date) }< tag pic need to update')

        serial_number_to_delete = list()
        # 遍历无效数据的信息，匹配原图路径
        for index in tqdm(range(len(tag_json_date))):
            update_record = 0
            tag_dict = tag_json_date[index]
            label_feature = tag_dict['labelFeature']
            original_filename = tag_dict['originalFileName']

            # 根据打回数据，得到对应原图路径
            image_name_list = original_filename.split('_')
            file_name = image_name_list[1]
            image_name = image_name_list[2]
            image_dir = os.path.join(data_path, file_name, image_name)

            # 遍历coco-json文件信息，匹配路径，把需要覆盖的图片找出来，并删掉所有图片与坐标信息
            for i in range(len(images_list)):
                images_info = images_list[i]
                original_dir = images_info['image_dir']
                if original_dir == image_dir:
                    update_record += 1
                    serial_number_to_delete.append(i)

        # dele_record = list(reversed(serial_number_to_delete))
        serial_number_to_delete.sort()
        dele_records = serial_number_to_delete[::-1]
        for i in dele_records:
            del json_data['images'][i]
            del json_data['annotations'][i]

    with open(json_save_dir, 'w') as fw:
        json.dump(json_data, fw)
        print(f"{json_save_dir} have succeed to write")





if __name__ == '__main__':
    main()