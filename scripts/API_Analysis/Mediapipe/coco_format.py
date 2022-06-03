'''
    调用Mediapipe，一次输入单张图片输出对应的手关键点并汇总到一个json文件里面，并且按照我们定义的crop image id 保存所有的cropimage
'''
import os
import cv2
import mediapipe as mp
from json_tools import make_json_head, convert_coco_format
import numpy as np
import json
from tqdm import tqdm
import copy

name = 'yongshong'
DATA_PATH = fr"G:\test_data\new_data\dataset\{name}\images"
IMAGE_SAVE_PATH = r"G:\test_data\new_data\crop_images"
ANNO_SAVE_PATH = fr"G:\test_data\new_data\dataset\{name}\anno"
JSON_NAME = fr'{name}.json'
coco_start_id = 1405646
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
    coco_id = coco_start_id

    is_exists = os.path.exists(IMAGE_SAVE_PATH)
    if not is_exists:
        os.makedirs(IMAGE_SAVE_PATH)
        print('path of %s is build' % IMAGE_SAVE_PATH)
    is_exists = os.path.exists(ANNO_SAVE_PATH)
    if not is_exists:
        os.makedirs(ANNO_SAVE_PATH)
        print('path of %s is build' % ANNO_SAVE_PATH)

    json_file = make_json_head()
    # # mp.solutions.drawing_utils用于绘制
    # mp_drawing = mp.solutions.drawing_utils
    #
    # # 参数：1、颜色，2、线条粗细，3、点的半径
    # DrawingSpec_point = mp_drawing.DrawingSpec((0, 255, 0), 2, 2)
    # DrawingSpec_line = mp_drawing.DrawingSpec((0, 0, 255), 2, 2)

    # mp.solutions.hands，是人的手
    mp_hands = mp.solutions.hands

    # 参数：1、是否检测静态图片(false:视频流)，2、手的数量，3、检测阈值，4、跟踪阈值
    hands_mode = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.2, min_tracking_confidence=0.2)

    # img_files = json_data(DATA_PATH, JSON_DIR)
    img_files = os.listdir(DATA_PATH)
    print(f"There are {len(img_files)} images")
    # len(img_files)
    for index in tqdm(range(len(img_files))):
        img_dir = os.path.join(DATA_PATH, img_files[index])
        img_name = os.path.split(img_dir)[1]
        image = cv2.imread(img_dir)

        # file = 'input.jpg'
        # image = cv2.imread(file)

        image_hight, image_width, _ = image.shape
        image1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 初始化坐标向量preds21*3， 置信度maxvals统一为0.5
        preds = np.zeros((num_joints, 3))
        maxvals = 0.5
        img_height, img_width, _ = image.shape

        # 处理RGB图像
        results = hands_mode.process(image1)
        # save the landsmarks
        landmarks_list = list()

        if not results.multi_hand_landmarks:
            value = False


        else:
            for hand_landmarks in results.multi_hand_landmarks:
                value = True
                for k in range(num_joints):
                    mp_w, mp_h = hand_landmarks.landmark[k].x * img_width, hand_landmarks.landmark[k].y * img_height
                    preds[k, 0] = mp_w
                    preds[k, 1] = mp_h
                    preds[k, 2] = maxvals
                landmarks_list.append(copy.deepcopy(preds))
                # print('hand_landmarks:', hand_landmarks)
                # print(
                #     f'Index finger tip coordinates: (',
                #     f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
                #     f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_hight})'
                # )
                #

                # mp_drawing.draw_landmarks(
                #     image, hand_landmarks, mp_hands.HAND_CONNECTIONS, DrawingSpec_point, DrawingSpec_line)

        for landmarks in landmarks_list:
            json_prefix = 'test2017'
            convert_coco_format(image, landmarks, json_file, json_prefix, IMAGE_SAVE_PATH, coco_id, img_name)
            coco_id += 1



        # save_dir = os.path.join(SAVE_PATH, img_name)
        # cv2.imwrite(save_dir, image)

    hands_mode.close()
    print(f'Next dataset cordID start from >> {coco_id} <<')
    json_path = os.path.join(ANNO_SAVE_PATH, JSON_NAME)
    with open(json_path, 'w') as fw:
        json.dump(json_file, fw)
        # print(f"{save_path} have succeed to write")

if __name__ == "__main__":
    main()