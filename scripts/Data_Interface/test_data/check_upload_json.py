import json
import os
import cv2
import numpy as np
from tools import draw_2d_points

json_path = r'E:\test_data\test_data_from_whole_body\upload_annotations'
json_files = os.listdir(json_path)
cv2.namedWindow('a', cv2.WINDOW_NORMAL)
count = 0
for file in json_files:
    if not file.endswith('.json'):
        continue
    json_dir = os.path.join(json_path, file)
    with open(json_dir, 'r')as f:
        json_data = json.load(f)
        image_info = json_data['images'][0]
        annotation_info = json_data['annotations'][0]
        image_dir = image_info['image_dir']
        keypoints_list = annotation_info['keypoints']

        img = cv2.imread(image_dir)
        for keypoints in keypoints_list:
            prieds = np.array(keypoints).reshape(21, 3)
            img = draw_2d_points(prieds, img)

        print(f"id：{image_info['id']}\tcount:{count}")
        count += 1
        cv2.imshow('a', img)
        cv2.waitKey(0)
