import copy
import json
import os
import shutil
import numpy as np
from tqdm import tqdm
import cv2
from json_tools import crop_box
from tools import draw_2d_points



badcase_txt = r'F:\Model_output\Output\coco\total_badcase_val.txt'
badcase_image_path = r'F:\image\coco'
# image_path = r'E:\Data\landmarks\HFB\halpe_Full-Body_Human_Keypoints_and_HOL-Det_dataset\hico_20160224_det\images'
image_path = r'F:\image\coco\val'

json_path = r'F:\image\coco\sort_coco_wholebody_val_v1.0.json'
# crop_json_path = r'E:\Data\landmarks\HFB\HFB\annotations\person_keypoints_train2017.json'
save_path = r'F:\image\coco\badcase'
# suspect_record_path = r'E:\Data\landmarks\HFB\halpe_Full-Body_Human_Keypoints_and_HOL-Det_dataset\suspect_image'

HandFig_Num = 21
CROP_FACTOR = 2.2
MIN_SIZE = 48

def check(landmarks, crop_landmarks, img):
    coco_kps = landmarks.copy()
    coco_kps[:, :2] = crop_landmarks[:, :2]
    coco_kps[:, 2] = landmarks[:, 2]
    kps_valid_bool = coco_kps[:, -1].astype(bool)
    coco_kps[~kps_valid_bool, :2] = 0
    img = draw_2d_points(coco_kps, img)
    cv2.imshow('show', img)
    cv2.waitKey(0)

def main():
    # cv2.namedWindow("show", cv2.WINDOW_NORMAL)

    save_file = os.path.split(badcase_txt)[1]
    for gg in range(10):
        save_dir = os.path.join(save_path, str(gg) + '_' + save_file)
        if os.path.exists(save_dir):
            continue
        else:
            break
    badcase_resave_dir = os.path.join(save_path, str(gg) + '_resave_' + save_file)

    have_count = 0
    count = 0

    # 得到所有badcase的路径
    badcase_image_dir_list = list()
    with open(badcase_txt, 'r')as f_txt:
        badcase_info = f_txt.readlines()

    for i in range(len(badcase_info)):
        badcase_image_dir = badcase_info[i].split('\n')[0]
        badcase_image_dir_list.append(badcase_image_dir)
        num = len(badcase_image_dir_list)


    # 得到全身的json数据
    print(f"loading the whole body json:{json_path}")
    with open(json_path, 'r') as whole_json:
        whole_json_data = json.load(whole_json)
        image_json_info = whole_json_data['images']
        annotations_json_info = whole_json_data['annotations']

    # 遍历badcase
    for badcase_dir in badcase_image_dir_list:
        badcase_image_dir = os.path.join(badcase_image_path, '.' + badcase_dir)
        print(
            f'''
 - - -- -- ---——————————————————————————————————--- -- -- - -
              {badcase_dir}
 - - -- -- ---——————————————————————————————————--- -- -- - -''')
        badcase_image = cv2.imread(badcase_image_dir)
        badcase_img_h, badcase_img_w = badcase_image.shape[:2]

        record_dir_list = list()
        record_res_list = list()
        record_index_list = list()
        # record_image_list = list()

        # 遍历whole body=>得到每张图片的crop image
        for index in tqdm(range(len(image_json_info))):

            file_name = image_json_info[index]['file_name']
            # mode = file_name.split('_')[1]
            # img_dir = os.path.join(image_path, mode, file_name)
            img_dir = os.path.join(image_path, file_name)
            img = cv2.imread(img_dir)

            assert image_json_info[index]['id'] == annotations_json_info[index]['image_id']
            # keypoints = np.array(annotations_json_info[index]['keypoints']).reshape(136, 3)

            for hand_index in range(2):
                hand_keypoints = np.zeros([HandFig_Num, 3])
                if hand_index == 0:
                    hand_keypoints = np.array(annotations_json_info[index]['lefthand_kpts']).reshape(21, 3)
                else:
                    hand_keypoints = np.array(annotations_json_info[index]['righthand_kpts']).reshape(21, 3)

                if np.all(hand_keypoints == 0):
                    continue
                crop_image_fromwhole, crop_landmarks = crop_box(img, hand_keypoints.copy(), box_factor=CROP_FACTOR)
                # crop_image_fromwhole = crop_image_fromwhole
                img_h, img_w = crop_image_fromwhole.shape[:2]

                # check the crop image
                # check(hand_keypoints, crop_landmarks, copy.deepcopy(crop_image_fromwhole))

                if (badcase_img_h, badcase_img_w) == (img_h, img_w):
                    res_img = crop_image_fromwhole.astype(np.float32) - badcase_image.astype(np.float32)
                    res_img = res_img / (img_w * img_h)

                    # res = np.abs(np.sum(res_img))
                    res = np.sum(np.abs(res_img))
                    if res < 50:
                        badcase_img_name = os.path.split(badcase_image_dir)[1]
                        print(f'{file_name}:{badcase_img_name} => {res}\t count:{count}-{have_count}-{num}')
                        record_dir_list.append(img_dir)
                        record_res_list.append(res)
                        record_index_list.append(index)
                        # record_image_list.append(np.hstack([crop_image_fromwhole, badcase_image]))

        have_count += 1
        if len(record_res_list) == 0:
            with open(badcase_resave_dir, 'a')as f:
                f.write(badcase_dir + "\n")
            print(f'There is no image to match\t count:{count}-{have_count}-{num}')
            continue

        res = min(record_res_list)
        i = record_res_list.index(res)
        img_dir = record_dir_list[i]
        index = record_index_list[i]
        # canve = record_image_list[i]
        image_json_info.pop(index)
        annotations_json_info.pop(index)
        print(f'{os.path.split(img_dir)[1]}<===========>{badcase_img_name}')
        with open(save_dir, 'a') as f:
            f.write(img_dir + "\n")
            # suspect_record_dir = os.path.join(suspect_record_path, str(res) + badcase_img_name)
            # cv2.imwrite(suspect_record_dir, canve)
        count += 1

    print(f"This is {len(badcase_image_dir_list)} badcase image")
    print(f'return {count}===>')


if __name__ == '__main__':
    main()
