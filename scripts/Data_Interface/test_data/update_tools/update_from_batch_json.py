from .tools import get_file_list, get_keypoints
import os
import json
from tqdm import tqdm


def update_from_batch_json(json_data, batch_sample_path, save_dir, debug=False):
    record_index = None
    # 按时间序列返回json文件
    json_files = get_file_list(batch_sample_path)
    for json_file in json_files:
        # 每次覆盖完，更新json信息
        images_list = json_data['images']

        batch_sample_dir = os.path.join(batch_sample_path, json_file)
        with open(batch_sample_dir, 'r', encoding='UTF-8')as f:
            batch_sample_data = json.load(f)
        print(f'There is >{ len(batch_sample_data) }< tag pic need to update')

        # 遍历打回数据的信息，匹配原图路径
        for index in tqdm(range(len(batch_sample_data))):
            update_record = 0
            tag_dict = batch_sample_data[index]
            label_feature = tag_dict['labelFeature']
            original_filename = tag_dict['originalFileName']

            # 根据打回数据，得到对应原图路径
            image_dir = original_filename
            same_img_flag = 0

            # 遍历coco-json文件信息，匹配路径，把需要覆盖的图片找出来，并删掉所有图片与坐标信息
            for i in range(len(images_list)):
                images_info = images_list[i]
                original_dir = images_info['image_dir']
                if original_dir == image_dir:
                    same_img_flag = 1
                    record_index = i
                    break

            if not same_img_flag:
                # print(f"There is not exist a same pic with {image_dir}")
                continue

            handlandmarks_list = get_keypoints(label_feature)
            json_data['annotations'][record_index]['keypoints'] = handlandmarks_list

    if debug:
        with open(save_dir, 'w') as fw:
            json.dump(json_data, fw)
            print("train2017.json have succeed to write")
