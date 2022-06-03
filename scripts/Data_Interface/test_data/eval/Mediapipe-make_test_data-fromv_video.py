import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

import sys
sys.path.append("..")
from convert_coco_format import convert_coco_format_from_wholebody

from google.protobuf import json_format
import numpy as np
import json
from json_tools import make_json_head
import os
from copy import deepcopy
from tools import draw_2d_points

NUM_JOINTSNUM = 21


def main():
    start_id = 1_400_000
    mode = 'mediapipe-full'
    save_dir = r'E:\test_data\test_data_from_whole_body\annotations\coco_eval\dt'
    total_save_dir = r'E:\test_data\test_data_from_whole_body\annotations'
    video_dir = r'E:\test_data\test_video'

    coco_id = 0
    image_id = start_id
    cv2.namedWindow('Window', cv2.WINDOW_NORMAL)
    json_file = make_json_head()
    json_total = deepcopy(json_file)

    for i in range(1, 11):
        json_unit = deepcopy(json_file)
        video_name = f'hand_test_{str(i).zfill(2)}.mp4'
        video_path = os.path.join(video_dir, video_name)
        cap = cv2.VideoCapture(video_path)
        with mp_hands.Hands(
                model_complexity=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                max_num_hands = 2) as hands:
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    # If loading a video, use 'break' instead of 'continue'.
                    break

                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                image.flags.writeable = False
                # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(image)

                # 初始化坐标向量preds21*3， 置信度maxvals统一为0.5
                preds = np.zeros((21, 3))
                maxvals = 2
                img_height, img_width, _ = image.shape
                landmarks = np.zeros([42, 3])

                if results.multi_hand_landmarks:
                    for index in range(len(results.multi_hand_landmarks)):
                        hand_landmarks = results.multi_hand_landmarks[index]
                        hand_type = results.multi_handedness[index]
                        hand_type = json_format.MessageToDict(hand_type)['classification'][0]['label']

                        for k in range(NUM_JOINTSNUM):
                            mp_w, mp_h = hand_landmarks.landmark[k].x * img_width, hand_landmarks.landmark[k].y * img_height
                            preds[k, 0] = mp_w
                            preds[k, 1] = mp_h
                            preds[k, 2] = maxvals

                            # json_total, json_file, img, coco_id, image_id, landmarks
                        flag = convert_coco_format_from_wholebody(json_total, json_unit, image, coco_id, image_id, preds)
                        if flag:
                            coco_id += 1

                        image = draw_2d_points(preds, image)

                cv2.imshow('Window', image)
                cv2.waitKey(1)

                image_id += 1

        cap.release()
        save_name = video_name.split('.')[0]
        save_path = os.path.join(save_dir, save_name, save_name + f'_{mode}.json')
        print(f"writing >>{save_path}<< ......")
        with open(save_path, 'w') as f:
            json.dump(json_file, f)
            print("Success!")

    save_path = os.path.join(total_save_dir, mode + '.json')
    print(f"writing >>{save_path}<< ......")
    with open(save_path, 'w') as f:
        json.dump(json_total, f)
        print("Success!")


if __name__ == "__main__":
    main()