import os
import cv2
import mediapipe as mp
from json_tools import make_json_head
import numpy as np
import json
from tqdm import tqdm

VEDIO_IMG_PATH = r"G:\test_data\vedio_images\images"
DATA_PATH = r"G:\test_data\images"
# SAVE_PATH = r"G:\test_data\vedio_images\mediapipe"
SAVE_PATH = r"G:\test_data\vedio_images\anno"
JSON_DIR = r"G:\transmission\anno\v2_2_json\person_keypoints_test2017.json"

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
    hands_mode = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0, min_tracking_confidence=0)

    # img_files = json_data(DATA_PATH, JSON_DIR)
    # print(f"There are {len(img_files)} images")
    #
    # # len(img_files)
    # for index in tqdm(range(len(img_files))):
    #     img_dir = img_files[index]
    #     img_name = os.path.split(img_dir)[1]

    img_list = os.listdir(VEDIO_IMG_PATH)
    for index in tqdm(range(len(img_list))):
        img_name = img_list[index]
        img_dir = os.path.join(VEDIO_IMG_PATH, img_name)


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
        if not results.multi_hand_landmarks:
            value = False
            keypoints = np.zeros(63)

        else:
            for hand_landmarks in results.multi_hand_landmarks:
                value = True
                for k in range(num_joints):
                    mp_w, mp_h = hand_landmarks.landmark[k].x * img_width, hand_landmarks.landmark[k].y * img_height
                    preds[k, 0] = mp_w
                    preds[k, 1] = mp_h
                    preds[k, 2] = maxvals
                # print('hand_landmarks:', hand_landmarks)
                # print(
                #     f'Index finger tip coordinates: (',
                #     f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
                #     f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_hight})'
                # )
                #

                # mp_drawing.draw_landmarks(
                #     image, hand_landmarks, mp_hands.HAND_CONNECTIONS, DrawingSpec_point, DrawingSpec_line)

        img_id = os.path.splitext(img_name)[0]
        image_dict = dict({
            'license': 1,
            'id': int(img_id),
            'file_name': img_name
        })

        keypoints = preds.flatten()
        anno_dict = dict({
            'num_keypoints': 21,
            'iscrowd': 0,
            'keypoints': list(keypoints),
            'category_id': 1,
            'hand_value': value,
            'id': img_id

        })

        json_file['images'].append(image_dict)
        json_file['annotations'].append(anno_dict)

        # save_dir = os.path.join(SAVE_PATH, img_name)
        # cv2.imwrite(save_dir, image)

    hands_mode.close()

    json_path = os.path.join(SAVE_PATH, f'testdata_mediapipe_img.json')
    with open(json_path, 'w') as fw:
        json.dump(json_file, fw)
        # print(f"{save_path} have succeed to write")

if __name__ == "__main__":
    main()