# 程序效果：
# 检查json文件中的关键点信息，show出gt中手势关键点到对应图片

import os
import sys
import cv2
import json
import numpy as np
from tqdm import tqdm

pro_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(pro_dir)

from tools import draw_2d_points


def main(json_path):
    cv2.namedWindow("aaa", cv2.WINDOW_NORMAL)

    with open(json_path, "r") as f:
        data = json.load(f)
        imgs_info = data['images']
        annos_info = data['annotations']
        #print(annos_info)

    num_imgs = len(imgs_info)
    # 生成0-num_imgs数字列表
    for i in range(num_imgs):
        # if i < 69500:
        #     continue
        img_info = imgs_info[i]
        anno_info = annos_info[i]

        # assert img_info['id'] == anno_info['image_id']

        if not 1400000 > anno_info['image_id'] >= 1300000:
            continue
        print(anno_info['image_id'])

        # file_name = img_info['file_name']
        # image_path = os.path.join(data_dir, file_name)
        image_path = img_info['image_dir']
        print(image_path)
        if not image_path == r'G:\\imgdate2\\HO3D_v3\\HO3D_v3\\train\\GSF14\\rgb\\0921.jpg'.replace(r'\\', '\\'):
            continue
        kp_points = np.asarray(anno_info['keypoints']).reshape((21, 3))

        # print(kp_points)
        img1 = cv2.imread(image_path)
        img2 = draw_2d_points(kp_points, img1, 21)

        if "hand_type" in anno_info.keys():
            hand_type = anno_info['hand_type']
            img2 = cv2.putText(img2, f'{hand_type}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)


        print(img_info['image_dir'])
        # if img_info['id'] == 'HICO_train2015_00002027':
        cv2.imshow("aaa", img2)
        # cv2.imwrite(save_dir, img2)
        if cv2.waitKey(0) == 27:
            exec("Esc clicked!")

    print(f'Hello')

if __name__ == "__main__":
    json_path_ = \
        r"E:\left_hand_label_data\annotations\person_keypoints_train2017.json"
    # data_dir_ = r'E:\left_hand_label_data\annotations\person_keypoints_val2017.json'

    main(json_path_)
