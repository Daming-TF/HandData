import json
import os
import cv2
import numpy as np
import copy

from library.tools import draw_2d_points

data_path = r'F:\image\CMU\hand_labels_synth\hand_labels_synth'
newjson_dir = r'E:\数据标记反馈\cmu-synth(6131)\批次样本\6131-手势关键点-2022_2_15-3-第一批.json'


cv2.namedWindow('left-new', cv2.WINDOW_NORMAL)
with open(newjson_dir, 'r', encoding='UTF-8') as f:
    json_labels = json.load(f)

    exit_flag = 0
    i = 0
    while 1:
        hands_landmarks = json_labels[i]['labelFeature']
        sampleID = json_labels[i]['sampleID']
        originalFileName = json_labels[i]['originalFileName']

        image_info = originalFileName.split('_')
        file_name = 'our_'+image_info[2]
        image_name = originalFileName[originalFileName.find(f'{file_name}')+len(file_name)+1:]

        image_dir = os.path.join(data_path, file_name, image_name)

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

        print(f'image name:{image_name}\tindex:{i}')
        print(f'sample ID:{sampleID}')
        canvas = np.hstack([newimage, image])

        if not image_dir == r'F:\image\CMU\hand_labels_synth\hand_labels_synth\our_train\2439.jpg':
            i += 1
            if i == len(json_labels) or exit_flag == 1:
                print('Finished checking all the data')
                break
            continue
        cv2.imshow('left---new', canvas)

        # # 检查旧数据是否修正
        # if sampleID in [51443339, 51443345, 51440137, 51440215, 51440287, 51440289, 51440683, 51440741, 51440799, 51441061, 51442379, 51442727, 51443073, 51448839]:
        #     print(f'sample ID:{sampleID}')
        #     cv2.imshow('left-new', canvas)
        #     cv2.waitKey(0)
        # i = i+1

        while 1:
            key = cv2.waitKey(1) & 0xFF
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
