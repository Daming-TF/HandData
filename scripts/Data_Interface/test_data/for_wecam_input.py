import os.path
import json
import cv2
import mediapipe as mp
import numpy as np
from google.protobuf import json_format
from convert_coco_format import convert_coco_format_from_wholebody
from json_tools import make_json_head

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

vedio_path = fr'E:\test_data\test_video'
save_path = r"E:\test_data\test_data_from_whole_body\annotations"
data_path = r'E:\test_data\test_data_from_whole_body\images'
JSON_NAME = f'mediapipe_lite-vedio.json'
vedio_names = ["hand_test_01", "hand_test_02", "hand_test_03", "hand_test_04", "hand_test_05", "hand_test_06",
                   "hand_test_07", "hand_test_08", "hand_test_09", "hand_test_10"]
# vedio_names = ["hand_test_08"]

start_id = 1_400_000    # 1_400_000
count = start_id


def main():
    json_file = make_json_head()
    count = start_id
    for vedio_name in vedio_names:
        vedio_dir = os.path.join(vedio_path, vedio_name+'.mp4')
        # For webcam input:
        cap = cv2.VideoCapture(vedio_dir)
        with mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:

            while cap.isOpened():
                success, image = cap.read()
                if not success:
                  print("Ignoring empty camera frame.")
                  # If loading a video, use 'break' instead of 'continue'.
                  break

                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(image)

                # Draw the hand annotations on the image.
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                preds = np.zeros((21, 3))
                maxvals = 2
                img_height, img_width, _ = image.shape
                landmarks = np.zeros([42, 3])

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:

                        for index in range(len(results.multi_hand_landmarks)):
                            hand_landmarks = results.multi_hand_landmarks[index]
                            hand_type = results.multi_handedness[index]
                            hand_type = json_format.MessageToDict(hand_type)['classification'][0]['label']

                            for k in range(21):
                                mp_w, mp_h = hand_landmarks.landmark[k].x * img_width, hand_landmarks.landmark[k].y * img_height
                                preds[k, 0] = mp_w
                                preds[k, 1] = mp_h
                                preds[k, 2] = maxvals

                            # 注意官网输入网络是需要翻折图片的，所以左右手位置和v3demo位置相反
                            if hand_type == 'Left':
                                landmarks[21:42] = preds
                            if hand_type == 'Right':
                                landmarks[0:21] = preds
                print(count)
                image = convert_coco_format_from_wholebody(image, landmarks, json_file, count, data_path)
                count += 1
                cv2.imshow('aa', image)
                cv2.waitKey(1)

            #     mp_drawing.draw_landmarks(
            #         image,
            #         hand_landmarks,
            #         mp_hands.HAND_CONNECTIONS,
            #         mp_drawing_styles.get_default_hand_landmarks_style(),
            #         mp_drawing_styles.get_default_hand_connections_style())
            #
            #     print()
            # # Flip the image horizontally for a selfie-view display.
            # cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
            # if cv2.waitKey(1) & 0xFF == 27:
            #   break
            cap.release()

    # json_path = os.path.join(save_path, JSON_NAME)
    # with open(json_path, 'w') as fw:
    #     json.dump(json_file, fw)
    #     print(f"{json_path} have succeed to write")


if __name__ == '__main__':
    main()