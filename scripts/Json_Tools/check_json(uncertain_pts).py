import os
import sys
import cv2
import json
import numpy as np

pro_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(pro_dir)

from vis_hand import NUM_HAND_KEYPOINTS
from  tools import draw_2d_points


pair = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20)
]


line_color = [(0, 215, 255), (0, 255, 204), (0, 134, 255), (0, 255, 50),
              (0, 255, 102), (77, 255, 222), (77, 196, 255), (77, 135, 255), (191, 255, 77), (77, 255, 77),
              (77, 191, 255), (204, 77, 255), (77, 222, 255), (255, 156, 127),
              (0, 127, 255), (255, 127, 77), (0, 77, 255), (255, 77, 36),
              (0, 77, 255), (0, 77, 255), (0, 77, 255), (0, 77, 255), (255, 156, 127), (255, 156, 127)]


# def draw_point( points, img_ori, radius=2, thickness=1):
#     img = img_ori.copy()
#
#     print(f'points shape: {points.shape}')
#
#     for i, point in enumerate(points):
#         x = int(point[0])
#         y = int(point[1])
#
#         if i == 0:
#             # 记录手腕关键点
#             rootx = x
#             rooty = y
#             prex = 0
#             prey = 0
#         if i == 1 or i == 5 or i == 9 or i == 13 or i == 17:
#             # 重设draw line的端点
#             prex = rootx
#             prey = rooty
#
#         # new add
#         if x == 0 and y == 0:
#             continue
#         # add new “if prex != 0 and prey != 0:” 是为了预防手腕关键点没有识别到？
#         if prex != 0 and prey != 0:
#             if (i > 0) and (i <= 4):
#                 cv2.line(img, (prex, prey), (x, y), (0, 0, 255), thickness, lineType=cv2.LINE_AA)
#                 cv2.circle(img, (x, y), radius, (0, 0, 255), -1)
#             if (i > 4) and (i <= 8):
#                 cv2.line(img, (prex, prey), (x, y), (0, 255, 255), thickness, lineType=cv2.LINE_AA)
#                 cv2.circle(img, (x, y), radius, (0, 255, 255), -1)
#             if (i > 8) and (i <= 12):
#                 cv2.line(img, (prex, prey), (x, y), (0, 255, 0), thickness, lineType=cv2.LINE_AA)
#                 cv2.circle(img, (x, y), radius, (0, 255, 0), -1)
#                 # putText(img, text, org, fontFace, fontScale, color, thickness=None, lineType=None, bottomLeftOrigin=None)
#             if (i > 12) and (i <= 16):
#                 cv2.line(img, (prex, prey), (x, y), (255, 255, 0), thickness, lineType=cv2.LINE_AA)
#                 cv2.circle(img, (x, y), radius, (255, 255, 0), -1)
#             if (i > 16) and (i <= 20):
#                 cv2.line(img, (prex, prey), (x, y), (255, 0, 0), thickness, lineType=cv2.LINE_AA)
#                 cv2.circle(img, (x, y), radius, (255, 0, 0), -1)
#
#             cv2.putText(img, text=str(i), org=(x, y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8,
#                         color=(0, 0, 0), thickness=2)
#
#         prex = x
#         prey = y
#
#     return img


# def draw_landmark(hand, img):
#     part_line = {}
#     show_img = img.copy()
#
#     if np.sum(hand[:, 2]) > NUM_HAND_KEYPOINTS:
#         for n in range(hand.shape[0]):
#             if hand[n, 2] == 0:
#                 continue
#
#             cor_x, cor_y = int(hand[n, 0]), int(hand[n, 1])
#             part_line[n] = (int(cor_x), int(cor_y))
#             cv2.circle(show_img, (int(cor_x), int(cor_y)), 1, (0, 0, 255), 2)
#
#         # Draw limbs
#         for i, (start_p, end_p) in enumerate(pair):
#             if start_p in part_line and end_p in part_line:
#                 start_xy = part_line[start_p]
#                 end_xy = part_line[end_p]
#                 cv2.line(show_img, start_xy, end_xy, line_color[i % int(len(pair) * 0.5)], 2)
#
#     return show_img


def main(img_folder, json_path, factor=4):
    with open(json_path, "r") as f:
        data = json.load(f)
        imgs_info = data['images']
        annos_info = data['annotations']

    num_imgs = len(imgs_info)
    for i in range(num_imgs):
        img_info = imgs_info[i]
        anno_info = annos_info[i]
        assert img_info['id'] == anno_info['id']

        img_path = os.path.join(img_folder, img_info['file_name'])
        # 将输入转换为数组并以21行3列形式展开，第三列数据表示图片状态（遮挡/正常）
        # 第三列数据为0：看不见不标； 为2看得见标
        kp_points = np.asarray(anno_info['keypoints']).reshape((21, 3))
        bbox = anno_info['bbox']
        x, y, h, w = bbox

        img = cv2.imread(img_path)
        if factor != 1:
            img = cv2.resize(img, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)
            kp_points = kp_points * factor
            x = x * factor
            y = y * factor
            h = h * factor
            w = w * factor

        img1 = draw_2d_points(kp_points, img)
        cv2.imshow('test', img1)

        img1 = cv2.rectangle(img1, (x, y), (x+w, y+h), (0, 0, 255), 3)


        print(str(img_info['file_name']))
        if str(img_info['file_name']) == '000000401765.jpg':
            print(kp_points)
        # cv2.imshow(f"show", img)
        if cv2.waitKey(0) == 27:
            exec("Esc clicked!")

    print(f'Hello')


if __name__ == "__main__":
    img_folder_ = r"D:\huya_AiBase\Project_hand\handpose\Hand2dDetection\imgdate\FH\FH\images\test2017"
    json_path_ = r"D:\huya_AiBase\Project_hand\handpose\Hand2dDetection\imgdate\FH\FH\annotations\person_keypoints_test2017.json"
    main(img_folder_, json_path_)
