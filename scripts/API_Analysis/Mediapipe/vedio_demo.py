import os.path
import numpy as np
import cv2
import mediapipe as mp
from tools import draw_2d_points


DATA_PATH = r"E:\test_data\test_video\hand_test_08.mp4"
SAVE_PATH = r"G:\test_data\vedio_images\mediapipe"
num_joints = 21

def main():
    # # mp.solutions.drawing_utils用于绘制
    # mp_drawing = mp.solutions.drawing_utils

    # # 参数：1、颜色，2、线条粗细，3、点的半径
    # DrawingSpec_point = mp_drawing.DrawingSpec((0, 255, 0), 1, 1)
    # DrawingSpec_line = mp_drawing.DrawingSpec((0, 0, 255), 1, 1)

    # mp.solutions.hands，是人的手
    mp_hands = mp.solutions.hands

    # 参数：1、是否检测静态图片，2、手的数量，3、检测阈值，4、跟踪阈值
    hands_mode = mp_hands.Hands(static_image_mode=True, max_num_hands=1,
                                min_detection_confidence=0, min_tracking_confidence=0)

    cap = cv2.VideoCapture(DATA_PATH)
    i = 0
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        image1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 处理RGB图像
        # 这里要输入RGB图片
        results = hands_mode.process(image1)

        # 初始化坐标向量preds21*3， 置信度maxvals统一为0.5
        preds = np.zeros((num_joints, 3))
        maxvals = 0.5
        img_height, img_width, _ = image.shape

        # 绘制
        # MULTI_HAND_LANDMARKS：表示为 21 个手部标志的列表
        # 每个标志由x、y和组成z。x和y分别[0.0, 1.0]由图像宽度和高度归一化
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for k in range(num_joints):
                    mp_w, mp_h = hand_landmarks.landmark[k].x * img_width, hand_landmarks.landmark[k].y * img_height
                    preds[k, 0] = mp_w
                    preds[k, 1] = mp_h
                    preds[k, 2] = maxvals
                # mp_drawing.draw_landmarks(
                #     image,
                #     hand_landmarks,
                #     mp_hands.HAND_CONNECTIONS,
                #     DrawingSpec_point,
                #     DrawingSpec_line)
            img = draw_2d_points(preds, image, 21)
        else:
            continue

        save_name = str(i).zfill(5) + '.jpg'
        save_path = os.path.join(SAVE_PATH, save_name)


        # cv2.imwrite(save_path, img)
        cv2.imshow('test', img)
        i += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    hands_mode.close()
    cv2.destroyAllWindows()
    cap.release()

if __name__ == "__main__":
    main()