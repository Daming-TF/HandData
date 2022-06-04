import os
import cv2
import numpy as np
from tqdm import tqdm
import json
from library.json_tools import make_json_head
from library.tools import draw_2d_points

json_path = r"E:\Data\landmarks\HFB\test\halpe_train_v1.json"
data_path1 = r'E:\Data\landmarks\HFB\halpe_Full-Body_Human_Keypoints_and_HOL-Det_dataset\hico_20160224_det\images\train2015'
data_path2 = r'E:\Data\landmarks\HFB\halpe_Full-Body_Human_Keypoints_and_HOL-Det_dataset\hico_20160224_det\images\test2015'

json_file = make_json_head()

with open(json_path, "r") as f:
    json_infos = json.load(f)
    img_list = json_infos['images']
    annotation_list = json_infos['annotations']

data_len = len(img_list)
for i in tqdm(range(data_len)):
    anno_info = annotation_list[i]
    landmarks = np.array(anno_info['keypoints']).reshape(136, 3)

    left_hand_keypoints = landmarks[94:115, :]
    right_hand_keypoints = landmarks[115:, :]

    if np.all(left_hand_keypoints[:, 0:2] == 0) and np.all(right_hand_keypoints[:, 0:2] == 0):
        continue

    id = anno_info["image_id"]

    for j in range(len(img_list)):
        img_info = img_list[j]
        if id == img_info['id']:
            file_name = img_info['file_name']
            img_dir = os.path.join(data_path1, file_name)
            if not os.path.exists(img_dir):
                img_dir = os.path.join(data_path2, str(id).zfill(12) + file_name)
            if not os.path.exists(img_dir):
                print('no image!')
                exit(0)
            img = cv2.imread(img_dir)
            img = draw_2d_points(left_hand_keypoints, img)
            img = draw_2d_points(right_hand_keypoints, img)
            cv2.imshow('show', img)
            cv2.waitKey(0)

            json_file['images'].append(img_info)
            json_file['annotations'].append(anno_info)
            img_list.pop(j)
            break

dir, filename = os.path.split(json_path)
save_dir = os.path.join(dir, "sort_"+filename)

with open(save_dir, 'w') as fw:
    json.dump(json_file, fw)
print(f'SUCCESS')
