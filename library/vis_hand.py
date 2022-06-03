import os
import sys
import json
import cv2
import numpy as np
from tqdm import tqdm

pro_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(pro_dir)

NUM_BODY_KEYPOINTS = 26
NUM_FACE_KEYPOINTS = 68
NUM_HAND_KEYPOINTS = 21

r_pair = [
    (94, 95), (95, 96), (96, 97), (97, 98), (94, 99), (99, 100), (100, 101), (101, 102), (94, 103), (103, 104), (104, 105),  # RightHand
    (105, 106), (94, 107), (107, 108), (108, 109), (109, 110), (94, 111), (111, 112), (112, 113), (113, 114),  # RightHand
    ]
l_pair = [
    (115, 116), (116, 117), (117, 118), (118, 119), (115, 120), (120, 121), (121, 122), (122, 123), (115, 124), (124, 125),  # LeftHand
    (125, 126), (126, 127), (115, 128), (128, 129), (129, 130), (130, 131), (115, 132), (132, 133), (133, 134), (134, 135)  # LeftHand
]


p_color = [(0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0),  # Nose, LEye, REye, LEar, REar
           (77, 255, 255), (77, 255, 204), (77, 204, 255), (191, 255, 77), (77, 191, 255), (191, 255, 77),  # LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist
           (204, 77, 255), (77, 255, 204), (191, 77, 255), (77, 255, 191), (127, 77, 255), (77, 255, 127),  # LHip, RHip, LKnee, Rknee, LAnkle, RAnkle, Neck
           (77, 255, 255), (0, 255, 255), (77, 204, 255),  # head, neck, shoulder
           (0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0), (77, 255, 255)] # foot


line_color = [(0, 215, 255), (0, 255, 204), (0, 134, 255), (0, 255, 50),
              (0, 255, 102), (77, 255, 222), (77, 196, 255), (77, 135, 255), (191, 255, 77), (77, 255, 77),
              (77, 191, 255), (204, 77, 255), (77, 222, 255), (255, 156, 127),
              (0, 127, 255), (255, 127, 77), (0, 77, 255), (255, 77, 36),
              (0, 77, 255), (0, 77, 255), (0, 77, 255), (0, 77, 255), (255, 156, 127), (255, 156, 127)]


def get_hand_keypoints(hand_type, kp_x, kp_y, kp_scores):
    hand = np.zeros((NUM_HAND_KEYPOINTS, 3), dtype=float)
    if hand_type == 0:  # right
        hand[:, 0] = kp_x[0, :].copy()
        hand[:, 1] = kp_y[0, :].copy()
        hand[:, 2] = kp_scores[0, :].copy()
    else:   # left
        hand[:, 0] = kp_x[1, :].copy()
        hand[:, 1] = kp_y[1, :].copy()
        hand[:, 2] = kp_scores[1, :].copy()
    return hand


def draw_landmark(hand, hand_type, img):
    part_line = {}
    show_img = img.copy()

    if np.sum(hand[:, 2]) > NUM_HAND_KEYPOINTS:
        # print(f'scores: {hand[:, 2]}')
        # print(f'x: {hand[:, 0]}')
        # print(f'y: {hand[:, 1]}')

        for n in range(hand.shape[0]):
            if hand[n, 2] == 0:
                continue

            cor_x, cor_y = int(hand[n, 0]), int(hand[n, 1])
            part_line[n] = (int(cor_x), int(cor_y))
            cv2.circle(show_img, (int(cor_x), int(cor_y)), 1, (0, 0, 255), 2)

        # Draw limbs
        for i, (start_p, end_p) in enumerate(l_pair if hand_type == 0 else r_pair):
            start_p -= (NUM_BODY_KEYPOINTS + NUM_FACE_KEYPOINTS)
            end_p -= (NUM_BODY_KEYPOINTS + NUM_FACE_KEYPOINTS)

            if start_p >= NUM_HAND_KEYPOINTS or end_p >= NUM_HAND_KEYPOINTS:
                start_p -= NUM_HAND_KEYPOINTS
                end_p -= NUM_HAND_KEYPOINTS

            if start_p in part_line and end_p in part_line:
                start_xy = part_line[start_p]
                end_xy = part_line[end_p]
                cv2.line(show_img, start_xy, end_xy, line_color[i % int(len(l_pair) * 0.5)], 2)

        cv2.imshow("show", show_img)
        if cv2.waitKey(0) == 27:
            exec("Esc clicked!")
        # return show_img


if __name__ == "__main__":
    bodyanno = json.load(open(r'D:\Data\landmarks\Halpe_Full-Body_Human_Keypoints_and_HOI-Det_dataset\halpe_val_v1.json'))
    image_folder = r'D:\Data\landmarks\Halpe_Full-Body_Human_Keypoints_and_HOI-Det_dataset\val2017'

    imgs = {}
    for img in bodyanno['images']:
        imgs[img['id']] = img

    for hidx, annot in enumerate(tqdm(bodyanno['annotations'])):
        if 'keypoints' in annot and type(annot['keypoints']) == list:
            imgname = str(imgs[annot['image_id']]['file_name'])

            if imgname != '000000037670.jpg':
                continue

            img = cv2.imread(os.path.join(image_folder, imgname))

            kp = np.array(annot['keypoints'])
            kp_x = np.asarray(kp[0::3][NUM_BODY_KEYPOINTS+NUM_FACE_KEYPOINTS:]).reshape(2, NUM_HAND_KEYPOINTS)
            kp_y = np.asarray(kp[1::3][NUM_BODY_KEYPOINTS+NUM_FACE_KEYPOINTS:]).reshape(2, NUM_HAND_KEYPOINTS)
            kp_scores = np.asarray(kp[2::3][NUM_BODY_KEYPOINTS+NUM_FACE_KEYPOINTS:]).reshape(2, NUM_HAND_KEYPOINTS)

            r_hand = get_hand_keypoints(0, kp_x, kp_y, kp_scores)       # right
            l_hand = get_hand_keypoints(1, kp_x, kp_y, kp_scores)      # left

            # Draw keypoints
            for hand_type, hand in enumerate([l_hand, r_hand]):
                draw_landmark(hand, hand_type, img)
