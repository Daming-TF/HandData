"""
gt数据里面，id是以image_id命名，所以该程序目的是重新命名id
"""
import json
from json_tools import make_json_head
from tqdm import tqdm

json_dir = r'E:\test_data\test_data_from_whole_body\annotations\average_pseudo_labels_update-coco.json'
save_dir = r'E:\test_data\test_data_from_whole_body\annotations\average_pseudo_labels_update-coco_id.json'

json_head = make_json_head()

with open(json_dir, 'r')as f:
    json_data = json.load(f)
    images = json_data['images']
    annotations = json_data['annotations']

    iter_num = len(images)

    for i in tqdm(range(iter_num)):
        image_info = images[i]

        annotation_info = annotations[i]
        annotation_info['id'] = i+1

        json_head['images'].append(image_info)
        json_head['annotations'].append(annotation_info)

    with open(save_dir, 'w')as sf:
        json.dump(json_head, sf)
        print("success!")

