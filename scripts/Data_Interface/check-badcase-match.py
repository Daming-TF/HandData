import cv2
import numpy as np

badcase_txt = r'E:\Data\landmarks\HFB\halpe_Full-Body_Human_Keypoints_and_HOL-Det_dataset\badcase\blocks_0_total_badcase_train.txt'

# 得到所有badcase的路径
badcase_image_dir_list = list()
with open(badcase_txt, 'r')as f_txt:
    badcase_info = f_txt.readlines()

for i in range(len(badcase_info)):
    badcase_image_dir = badcase_info[i].split('\n')[0]
    badcase_image_dir_list.append(badcase_image_dir)
    num = len(badcase_image_dir_list)

for index in range(num):
    badcase_record = badcase_image_dir_list[index]
    original_img_dir = badcase_record.split('**')[0]
    crop_img_dir = badcase_record.split('**')[1]

    im1 = cv2.imread(original_img_dir)
    im1 = cv2.resize(im1, (400, 400), interpolation=cv2.INTER_LINEAR)

    im2 = cv2.imread(crop_img_dir)
    im2 = cv2.resize(im2, (400, 400), interpolation=cv2.INTER_LINEAR)

    canve = np.hstack([im1, im2])
    cv2.imshow('show', canve)
    cv2.waitKey(0)
