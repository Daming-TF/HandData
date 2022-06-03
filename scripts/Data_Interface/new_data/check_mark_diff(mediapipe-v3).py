import json
import os
import cv2
import numpy as np
from tools import draw_2d_points
import copy
from json_tools import crop_box

# data_path = r'E:\test_data\test_data_from_whole_body\test\imgs'
newjson_dir = r'E:\test_data\test_data_from_whole_body\record\average_pseudo_labels_wo_badcase.json'
save_record_path = r'E:\left_hand_label_data\record'
save_invalid_path = r'E:\left_hand_label_data\invalid'
badcase_txt_name = r'badcase.txt'

for path in [save_invalid_path, save_record_path]:
    if not os.path.exists(path):
        os.mkdir(path)

cv2.namedWindow('show', cv2.WINDOW_NORMAL)
with open(newjson_dir, 'r') as f:
    json_labels = json.load(f)
    images = json_labels['images']
    annotations = json_labels['annotations']

    exit_flag = 0
    i = 1300
    while 1:
        image_info = images[i]
        annotations_info = annotations[i]
        image_dir = image_info['image_dir']
        keypoints = annotations_info['keypoints']

        image = cv2.imread(image_dir)
        image1 = copy.deepcopy(image)
        # for prieds in keypoints:
        image1 = draw_2d_points(np.array(keypoints).reshape([21, 3]), image1)

        print(f'image name:{image_dir}\t index:{i}')
        canvas = np.hstack([image1, image])
        cv2.imshow('show', canvas)

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
                txt_path = os.path.join(save_record_path, badcase_txt_name)
                img_examine_path = os.path.join(save_record_path, os.path.basename(image_dir))
                if not os.path.exists(img_examine_path):
                    cv2.imwrite(img_examine_path, image1)
                    with open(txt_path, 'a') as f:
                        f.write(image_dir+'\n')

                else:
                    print(f"({os.path.basename(image_dir)})has beeen recored")

                i = i + 1
                break

            # 无效样本
            elif key == ord('p'):
                # 检查图片是否已经标记
                txt_path = os.path.join(save_invalid_path, badcase_txt_name)
                img_examine_path = os.path.join(save_invalid_path, os.path.basename(image_dir))
                if not os.path.exists(img_examine_path):
                    cv2.imwrite(img_examine_path, image1)
                    with open(txt_path, 'a') as f:
                        f.write(image_dir+'\n')

                else:
                    print(f"({os.path.basename(image_dir)})has beeen recored")

                i = i + 1
                break

            elif key == ord('q'):
                exit_flag = 1
                break

        if i == len(images) or exit_flag == 1:
            print('Finished checking all the data')
            break