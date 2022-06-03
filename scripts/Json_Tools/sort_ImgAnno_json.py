import os
from tqdm import tqdm
import json
from json_tools import make_json_head

json_path = r"F:\image\coco\coco_wholebody_train_v1.0.json"

json_file = make_json_head()

with open(json_path, "r") as f:
    json_infos = json.load(f)
    img_list = json_infos['images']
    annotation_list = json_infos['annotations']

data_len = len(img_list)
for i in tqdm(range(data_len)):
    anno_info = annotation_list[i]

    if not anno_info["lefthand_valid"] and anno_info["righthand_valid"]:
        continue
    id = anno_info["image_id"]
    for j in range(data_len):
        img_info = img_list[j]
        if id == img_info['id']:
            json_file['images'].append(img_info)
            json_file['annotations'].append(anno_info)

dir, filename = os.path.split(json_path)
save_dir = os.path.join(dir, "sort_"+filename)

with open(save_dir, 'w') as fw:
    json.dump(json_file, fw)
print(f'SUCCESS')