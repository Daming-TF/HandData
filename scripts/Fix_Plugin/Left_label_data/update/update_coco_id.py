import json

from json_tools import make_json_head

json_path = r'E:\left_hand_label_data\annotations\person_keypoints_train2017.json'
save_path = r'E:\left_hand_label_data\annotations\id_update\person_keypoints_train2017.json'

json_head = make_json_head()

coco_id = 15640
with open(json_path, 'r')as f:
    json_data = json.load(f)
    images = json_data['images']
    annotations = json_data['annotations']

    for annotation in annotations:
        annotation['id'] = coco_id
        coco_id += 1

    print(coco_id)
    json_head['images'] = images
    json_head['annotations'] = annotations

with open(save_path, 'w')as sf:
    json.dump(json_head, sf)

