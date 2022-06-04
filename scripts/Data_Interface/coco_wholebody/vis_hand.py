"""
功能：原始图片可视化
"""

import json
import numpy as np
import cv2
import os
from library.tools import draw_2d_points


def main(img_path, json_path):
    with open(json_path, "r") as f:
        json_infos = json.load(f)
        img_list = json_infos['images']
        annotation_list = json_infos['annotations']

    for i in range(len(img_list)):
        img_info = img_list[i]
        anno_info = annotation_list[i]
        print(img_info['id'])
        print(anno_info['image_id'])
        assert img_info['id'] == anno_info['image_id']

        img_name = img_info['file_name']
        img_dir = os.path.join(img_path, img_name)
        image = cv2.imread(img_dir)

        hands_kpts = []
        if anno_info["lefthand_valid"]:
            hands_kpts.append(anno_info["lefthand_kpts"])
        if anno_info["righthand_valid"]:
            hands_kpts.append(anno_info["righthand_kpts"])
        # hands_kpts = []
        # kpts_list = []
        #
        # for mode in ['left', 'right']:
        #     exec("kpts_list = {}hand_kpts_list[i]".format(mode))
        #     if any(kpts_list):
        #         hands_kpts.append(kpts_list)

        for hand_kpts in hands_kpts:
            kp = np.array(hand_kpts).reshape(21,3)
            kp = kp[:, :2]
            print(kp)
            im = draw_2d_points(kp, image, 21 )
            # im = cv2.resize(im, (500, 500), interpolation=cv2.INTER_LINEAR)
            cv2.imshow("check gt", im)
            if cv2.waitKey(0) == 27:
                    exec("Esc clicked!")


if __name__ == "__main__":
    img_path = r"F:\image\coco\train2017\train2017"
    json_path = r"F:\image\coco\sort_coco_wholebody_train_v1.0.json"

    main(img_path, json_path)