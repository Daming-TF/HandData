import cv2
import numpy as np
import os
import json
from json_tools import make_json_head, convert_dict_to_list_landmarks
from tqdm import tqdm

json_dir = r'E:\数据标记反馈\6945\6945-手势关键点-2022_3_12—2041张待标注.json'
save_dir = r'E:\test_data\test_data_from_whole_body\annotations\need_to_remark.json'


def main():
    json_head = make_json_head()
    with open(json_dir, 'r')as f:
        json_data_list = json.load(f)

    count = 0
    for i in tqdm(range(len(json_data_list))):
        json_dict_unit = json_data_list[i]
        label_feature = json_dict_unit['labelFeature']
        original_filename = json_dict_unit['originalFileName']

        image = cv2.imread(original_filename)
        file_name = os.path.basename(original_filename)
        img_h, img_w = image.shape[:2]
        id = int(file_name.split('.')[0])
        landmarks = convert_dict_to_list_landmarks(image, label_feature)

        if id >= 1414398 and id <=1415085:
            count += 1
            print(count)
            continue

        image_dict = dict({
            'license': 1,
            'file_name': file_name,
            'image_dir': original_filename,
            'coco_url': 'Unavailable',
            'height': img_h,
            'width': img_w,
            'date_captured': '',
            'flickr_url': 'Unavailable',
            'id': id
        })

        anno_dict = dict({
            'segmentation': [],
            'num_keypoints': 21,
            'area': '',
            'iscrowd': 0,
            'keypoints': landmarks,
            'image_id': id,
            'bbox': [],  # 1.5 expand
            'category_id': 1,
            'id': id
        })

        json_head['images'].append(image_dict)
        json_head['annotations'].append(anno_dict)
    with open(save_dir, 'w')as f:
        json.dump(json_head, f)

if __name__ == '__main__':
    main()
