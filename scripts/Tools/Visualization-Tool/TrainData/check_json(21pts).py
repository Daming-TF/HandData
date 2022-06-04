# 程序效果：
# 检查json文件中的关键点信息，show出gt中手势关键点到对应图片

import os
import sys
import cv2
import json
import numpy as np

pro_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(pro_dir)

from tools import draw_2d_points

# save_path = r'G:\test_data\vedio_images\mediapipe_image'

def main(img_folder, json_path):
    cv2.namedWindow('Show', cv2.WINDOW_NORMAL)
    with open(json_path, "r") as f:
        data = json.load(f)
        # 该列表每个元素皆为一个img信息数据字典
        # 包括'license', 'file_name', 'coco_url', 'height', 'width', 'data_captured', 'flickr_url', 'id'
        imgs_info = data['images']
        # print(imgs_info)
        # annos_info数据包括'segmentation', 'num_keypoints', 'image_id', 'bbox', 'category_id', 'id',
        annos_info = data['annotations']
        #print(annos_info)

    num_imgs = len(imgs_info)
    # 生成0-num_imgs数字列表
    for i in range(num_imgs):
        # if i < 1400:
        #     continue
        img_info = imgs_info[i]
        anno_info = annos_info[i]
        # img_info['id'] ！= anno_info['id']情况触发异常
        print(img_info['id'])
        id = img_info['id']
        assert int(img_info['id']) == int(anno_info['id'])

        # if not img_info['file_name'] == '000000300212.jpg':
        #     continue

        img_path = os.path.join(img_folder, img_info['file_name'])
        # 转化的原因：列表创建指针数=元素数  增加内存和cpu消耗，对于存数字操作一般转化为数组
        # 将输入转换为数组并以21行3列形式展开，第三列数据表示图片状态（遮挡/正常）
        if np.all(np.asarray(anno_info['keypoints']) == 0):
            kp_points = np.zeros(63).reshape(21, 3)
        else:
            kp_points = np.asarray(anno_info['keypoints']).reshape((21, 3))

        # print(kp_points)
        img = cv2.imread(img_path)
        img = draw_2d_points(kp_points, img, 21)

        # bbox是四个坐标数据，如第一帧图片输出数据为(84, 84, 362, 362)
        if 'bbox' in anno_info.keys ():
            bbox = anno_info['bbox']
            x, y, h, w = bbox

            # 在图像上绘制一个简单的矩形
            img = cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 1)

        # img = cv2.resize(img, (600, 600), interpolation=cv2.INTER_LINEAR)

        # save_dir = os.path.join(save_path, img_info['file_name'])
        cv2.imshow("Show", img)
        # cv2.imwrite(save_dir, img)
        if cv2.waitKey(0) == 27:
            exec("Esc clicked!")

    print(f'Hello')

if __name__ == "__main__":
    img_folder_ = r"E:\Data\landmarks\YouTube3D\YouTube3D-crop2\images\train2017"
    # G:\test_data\debug\person_keypoints_test2017.json
    json_path_ = \
        r"E:\Data\landmarks\YouTube3D\YouTube3D-crop2\annotations\val2017-update-for-invaliddata.json"
    main(img_folder_, json_path_)
