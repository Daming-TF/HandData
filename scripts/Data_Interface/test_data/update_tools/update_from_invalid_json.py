import json
import os
from tqdm import tqdm

from .tools import get_file_list

def update_from_invalid_json(json_data, invalid_sample_path, json_save_dir):
    # 按时间序列返回json文件
    json_files = get_file_list(invalid_sample_path)
    print(f">> {len(json_files)} is invalid <<")
    for json_file in json_files:
        # 每次覆盖完，更新json信息
        images_list = json_data['images']

        invalid_sample_dir = os.path.join(invalid_sample_path, json_file)
        with open(invalid_sample_dir, 'r', encoding='UTF-8')as f:
            invalid_sample_date = json.load(f)
        print(f'There is >{ len(invalid_sample_date) }< tag pic need to update')

        serial_number_to_delete = list()
        # 遍历无效数据的信息，匹配原图路径
        for index in tqdm(range(len(invalid_sample_date))):
            update_record = 0
            tag_dict = invalid_sample_date[index]
            # label_feature = tag_dict['labelFeature']
            original_filename = tag_dict['originalFileName']

            # 根据打回数据，得到对应原图路径
            image_dir = original_filename

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