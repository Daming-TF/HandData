import json
import os
from tqdm import tqdm
from .convert_tools import get_file_list
from library.json_tools import get_ids, write_json

data_path = r'F:\image\Rendered Handpose Dataset Dataset\RHD\RHD_published_v2'


def update_from_invalid_json(images_dict, annotations_dict, invalid_sample_path, json_save_path):
    counter = 0
    ids = get_ids(images_dict)
    # sort the invalid json files according to time
    json_files = get_file_list(invalid_sample_path)
    for json_file in json_files:
        invalid_sample_dir = os.path.join(invalid_sample_path, json_file)
        with open(invalid_sample_dir, 'r', encoding='UTF-8')as f:
            invalid_sample_date = json.load(f)
        print(f'There is >{ len(invalid_sample_date) }< tag pic need to update')

        serial_number_to_delete = list()

        # Traverse the information of the returned data and match the original image path
        for index in tqdm(range(len(invalid_sample_date))):
            tag_dict = invalid_sample_date[index]
            original_filename = tag_dict['originalFileName']

            # extract the image path
            image_info = original_filename.split('_')
            file_name1 = image_info[1]
            file_name2 = image_info[2]
            image_name = image_info[3]
            image_dir = os.path.join(data_path, file_name1, file_name2, image_name)

            for image_id in ids:
                image_info = images_dict[image_id]
                original_dir = image_info['image_dir']
                if original_dir == image_dir:
                    serial_number_to_delete.append(image_id)
                    record_match(image_id, index, json_save_path, counter)
                    counter += 1

        serial_number_to_delete.sort()
        dele_records = serial_number_to_delete[::-1]
        for image_id in dele_records:
            annotations_dict.pop(image_id)
            images_dict.pop(image_id)

    write_json(images_dict, annotations_dict, json_save_path)


def record_match(image_id, index, save_path, counter):
    mode = None
    for keyword in ['train', 'val']:
        if keyword in save_path:
            mode = keyword
            break

    save_dir = os.path.join(os.path.split(save_path)[0], f'{mode}_invalid_data')
    save_path = os.path.join(save_dir, 'record.txt')
    os.makedirs(save_dir, exist_ok=True)
    if counter == 0:
        open(save_path, 'w').close()
    with open(save_path, 'a')as f:
        f.write(f"{image_id}**{index}\n")
