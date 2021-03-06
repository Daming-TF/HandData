"""
    查看重标数据是否合格，并记录
"""
import json
import os
import cv2
import numpy as np
import copy

from library.tools import draw_2d_points

data_path = r'G:\test_data\hardcase_data\dataset'
newjson_dir = r'E:\数据标记反馈\2月25日无效样本\5479-手势关键点-2022_1_25.json'


cv2.namedWindow('left-new', cv2.WINDOW_NORMAL)
with open(newjson_dir, 'r', encoding='UTF-8') as f:
    json_labels = json.load(f)

    exit_flag = 0
    i = 0
    while 1:
        hands_landmarks = json_labels[i]['labelFeature']
        sampleID = json_labels[i]['sampleID']

        image_name = json_labels[i]['originalFileName'].split('_')[1]
        json_name = os.path.splitext(image_name)[0] + '.json'
        data_name = json_labels[i]['originalFileName'].split('_')[0]

        image_dir = os.path.join(data_path, data_name, 'images', image_name)
        image = cv2.imread(image_dir)

        # 画新数据的关键点
        hand1 = np.zeros((21, 3))
        hand2 = np.zeros((21, 3))
        keys = hands_landmarks[0].keys()
        for key in keys:
            hand_info = key.split('-')
            hand_index = int(hand_info[0])
            landmark_index = int(hand_info[1])
            if hand_index == 0:
                hand1[landmark_index] = np.array(hands_landmarks[0][key])
            elif hand_index == 1:
                hand2[landmark_index] = np.array(hands_landmarks[0][key])

        newimage = draw_2d_points(hand1, copy.deepcopy(image))
        newimage = draw_2d_points(hand2, newimage)

        # cv2.imshow('show', newimage)
        # cv2.waitKey(0)

        # # 把旧json中的关键点打印出来
        # pried = np.zeros([21, 3])
        # with open(json_dir,'r')as of:
        #     old_json_data = json.load(of)
        #     annotations = old_json_data['annotations'][0]
        #     keypoints_list = annotations['keypoints']
        #     for keypoint in keypoints_list:
        #         pried[:, :2] = np.array(keypoint).reshape(21, 2)
        #         image = draw_2d_points(pried, image)

        # cv2.imshow('show', image)
        # cv2.waitKey(0)

        print(f'image name:{image_name}\tindex:{i}')
        print(f'sample ID:{sampleID}')
        canvas = np.hstack([newimage, image])

        cv2.imshow('la', canvas)

        while 1:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('f'):
                i = i + 1
                break
            elif key == ord('b'):
                i = i - 1
                break
            elif key == ord('m'):
                # 检查图片是否已经标记
                text_path = os.path.join(os.path.split(newjson_dir)[0],
                                         os.path.splitext(os.path.split(newjson_dir)[1])[0])
                # print('test!' + text_path)
                img_examine_path = os.path.join('E:\Questionable data feedback\image', str(sampleID) + '.jpg')
                if not os.path.exists(text_path):
                    os.mkdir(text_path)
                badcase_path = os.path.join(text_path, 'badcase.txt')
                if not os.path.exists(img_examine_path):
                    print('write image')
                    cv2.imwrite(img_examine_path, newimage)
                    print(img_examine_path)
                    print(image_name + f"({sampleID})")

                    with open(badcase_path, 'a') as f:
                        f.write(image_dir + f"\t({sampleID})\n")

                else:
                    print(image_name + f"({sampleID})has beeen recored")

                i = i + 1
                break

            elif key == ord('q'):
                exit_flag = 1
                break

        if i == len(json_labels) or exit_flag == 1:
            print('Finished checking all the data')
            break
