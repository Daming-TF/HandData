import json
import os
import numpy as np
import cv2
from library.tools import draw_2d_points

json_path = r'G:\test_data\hardcase_data\dataset\aiyu\anno'
data_path = r'G:\test_data\hardcase_data\dataset\aiyu\images'

json_names = os.listdir(json_path)
for json_name in json_names:
    json_dir = os.path.join(json_path, json_name)
    with open(json_dir, 'r')as f:
        json_data = json.load(f)
        keypoints_list = json_data['annotations'][0]['keypoints']
        print(f'{json_name}:{len(keypoints_list)}')
        if keypoints_list == []:
            continue
        keypoints = np.array(keypoints_list[0]).reshape(21, 2)
        img_dir = os.path.join(data_path, os.path.splitext(json_name)[0]+'.jpg')
        img = cv2.imread(img_dir)

        img1 = draw_2d_points(keypoints, img)

        cv2.imshow('test', img1)
        cv2.waitKey(0)
