'''
    调用Mediapipe，一次输入单张图片输出对应的手关键点并汇总到一个json文件里面
'''
import os
import cv2
import mediapipe as mp
from json_tools import make_json_head
import numpy as np
import json
from tqdm import tqdm
import time
from convert_coco_format import convert_coco_format_from_wholebody
from google.protobuf import json_format

DATA_PATH = r"C:\Users\Administrator\Pictures\Camera Roll"
SAVE_PATH = r"E:\left_hand_label_data"
JSON_NAME = f'mediapipe-test.json'
# JSON_DIR = r"G:\transmission\anno\v2_2_json\person_keypoints_test2017.json"

num_joints = 21

def json_data(data_dir, json_dir):
    img_dirs = []
    with open(json_dir, "r") as f:
        test_data = json.load(f)
    # check number of items
    img_info_list = test_data['images']
    anno_info_list = test_data['annotations']

    num_len = len(img_info_list)
    for i in tqdm(range(num_len)):
        img_info = img_info_list[i]
        anno_info = anno_info_list[i]
        assert img_info["id"] == anno_info["id"]

        img_name = img_info["file_name"]
        img_dir = os.path.join(data_dir, img_name)
        img_dirs.append(img_dir)

    return img_dirs

def main():
    i, j = 0, 0
    cv2.namedWindow("aa", cv2.WINDOW_NORMAL)
    is_exists = os.path.exists(SAVE_PATH)
    if not is_exists:
        os.makedirs(SAVE_PATH)
        print('path of %s is build' % SAVE_PATH)

    json_file = make_json_head()
    # mp.solutions.hands，是人的手
    mp_hands = mp.solutions.hands

    # 参数：1、是否检测静态图片(false:视频流)，2、手的数量，3、检测阈值，4、跟踪阈值
    hands_mode = mp_hands.Hands(model_complexity=1, static_image_mode=True, max_num_hands=2,
                                min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # img_files = json_data(DATA_PATH, JSON_DIR)
    img_files = os.listdir(DATA_PATH)
    print(f"There are {len(img_files)} images")
    # len(img_files)
    total_time = float(0)


    count = 0
    for index in tqdm(range(len(img_files))):
        count += 1
        # if count < 47:
        #     continue
        # if count > 5:
        #     break
        id = int(img_files[index].split('.')[0])
        img_dir = os.path.join(DATA_PATH, img_files[index])
        image = cv2.imread(img_dir)

        image_hight, image_width, _ = image.shape
        image1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 初始化坐标向量preds21*3， 置信度maxvals统一为0.5
        preds = np.zeros((num_joints, 3))
        maxvals = 0.5
        img_height, img_width, _ = image.shape

        # 处理RGB图像

        results = hands_mode.process(image1)

        landmarks = np.zeros([42, 3])

        if results.multi_hand_landmarks:
            for index in range(len(results.multi_hand_landmarks)):
                hand_landmarks = results.multi_hand_landmarks[index]
                hand_type = results.multi_handedness[index]
                hand_type = json_format.MessageToDict(hand_type)['classification'][0]['label']

                for k in range(num_joints):
                    mp_w, mp_h = hand_landmarks.landmark[k].x * img_width, hand_landmarks.landmark[k].y * img_height
                    preds[k, 0] = mp_w
                    preds[k, 1] = mp_h
                    preds[k, 2] = maxvals

                # 注意官网输入网络是需要翻折图片的，所以左右手位置和v3demo位置相反
                if hand_type == 'Left':
                    landmarks[21:42] = preds
                if hand_type == 'Right':
                    landmarks[0:21] = preds

        image = convert_coco_format_from_wholebody(image, landmarks, json_file, id, DATA_PATH)
        cv2.imshow('aa', image)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('m'):
            i = i + 1
        if key == ord('p'):
            j = j+ 1

    hands_mode.close()

    # avg_time = total_time / count
    # FPS = 1000 / avg_time
    # print(f'Avage time is:{avg_time}')
    # print(f'FPS is:{FPS}')
    json_path = os.path.join(SAVE_PATH, JSON_NAME)

    # # 写入json文件
    # with open(json_path, 'w') as fw:
    #     json.dump(json_file, fw)
    #
    #     print(f"{json_path} have succeed to write")

    print(f'i:{i}\tj:{j}')
if __name__ == "__main__":
    main()