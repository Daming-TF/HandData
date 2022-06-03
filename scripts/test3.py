import json
from tqdm import tqdm
from models.json_tools import make_json_head

original_json_dir = r'G:\test_data\new_data\new_data_from_whole_body\annotations\person_keypoints_test2017-update.json'
now_json_dir = r'E:\left_hand_label_data\annotations\person_keypoints_test2017.json'
save_dir = r'E:\left_hand_label_data\annotations\original_whole_update\missing_data.json'

json_head = make_json_head()

with open(original_json_dir, 'r')as f:
    original_json_data = json.load(f)
    original_images = original_json_data['images']
    original_annottaions = original_json_data['annotations']
    origenal_iter_num = len(original_images)

with open(now_json_dir, 'r')as f:
    now_json_data = json.load(f)
    now_images = now_json_data['images']
    now_annotations = now_json_data['annotations']
    now_iter_num = len(now_images)

for i in tqdm(range(origenal_iter_num)):
    flag = 0
    original_image_info = original_images[i]
    original_annottaion_info = original_annottaions[i]
    image_id = original_annottaion_info['image_id']
    for i in tqdm(range(now_iter_num)):
        now_annottaion_info = now_annotations[i]
        now_image_id = now_annottaion_info['image_id']
        if image_id == now_image_id:
            flag = 1
            break

    if not flag:
        json_head['images'].append(original_image_info)
        json_head['annotations'].append(original_annottaion_info)

print(f"writing the '{save_dir}'......")
with open(save_dir, 'w')as f:
    json.dump(json_head, f)