import os
import sys
import cv2
import pickle
import numpy as np
from scripts.load_db import load_dataset

pro_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(pro_dir)



# FINGERTIP_IDXS_MANO = [744, 320, 443, 554, 671]
# FINGERTIP_IDXS_MANO = [744, 320, 443, 671, 554]
FINGERTIP_IDXS_MANO = [745, 333, 444, 672, 555]
FINGERTIP_IDXS = [4, 8, 12, 16, 20]
NUM_JOINTS = 21


def landmarks_mappling(landmarks, finger_tips):
    # rearrange index according to the standard
    new_landmarks = np.zeros((NUM_JOINTS, 3))

    new_landmarks[0] = landmarks[0]

    new_landmarks[1] = landmarks[13]
    new_landmarks[2] = landmarks[14]
    new_landmarks[3] = landmarks[15]
    new_landmarks[4] = finger_tips[0]

    new_landmarks[5] = landmarks[1]
    new_landmarks[6] = landmarks[2]
    new_landmarks[7] = landmarks[3]
    new_landmarks[8] = finger_tips[1]

    new_landmarks[9] = landmarks[4]
    new_landmarks[10] = landmarks[5]
    new_landmarks[11] = landmarks[6]
    new_landmarks[12] = finger_tips[2]

    new_landmarks[13] = landmarks[10]
    new_landmarks[14] = landmarks[11]
    new_landmarks[15] = landmarks[12]
    new_landmarks[16] = finger_tips[4]

    new_landmarks[17] = landmarks[7]
    new_landmarks[18] = landmarks[8]
    new_landmarks[19] = landmarks[9]
    new_landmarks[20] = finger_tips[3]

    return new_landmarks


def _init_mano(mano_left_path, mano_right_path, verbose=True):
    with open(mano_left_path, 'rb') as lhand_file:
        lhand_data = pickle.load(lhand_file, encoding='latin1')

    with open(mano_right_path, 'rb') as rhand_file:
        rhand_data = pickle.load(rhand_file, encoding='latin1')

    if verbose:
        print(" [!] Check Key and Value in MANO model")
    for key, value in lhand_data.items():
        if isinstance(value, str):
            if verbose:
                print(f'{key}:, {value} - value')
        else:
            if verbose:
                print(f'{key}, {value.shape} - shape')

    lhand_reg = lhand_data['J_regressor'].toarray()  # from csc_matrix to numpy array
    rhand_reg = rhand_data['J_regressor'].toarray()  # from csc_matrix to numpy array

    return lhand_reg, rhand_reg


def draw_points(points, im):
    for i in range(NUM_JOINTS):
        point = points[i]
        x = int(point[0])
        y = int(point[1])

        if i == 0:
            rootx = x
            rooty = y
        if i == 1 or i == 5 or i == 9 or i == 13 or i == 17:
            prex = rootx
            prey = rooty

        if (i > 0) and (i <= 4):
            cv2.line(im, (prex, prey), (x, y), (0, 0, 255), 3, lineType=cv2.LINE_AA)
            cv2.circle(im, (x, y), 5, (0, 0, 255), -1)
        if (i > 4) and (i <= 8):
            cv2.line(im, (prex, prey), (x, y), (0, 255, 255), 3, lineType=cv2.LINE_AA)
            cv2.circle(im, (x, y), 5, (0, 255, 255), -1)
        if (i > 8) and (i <= 12):
            cv2.line(im, (prex, prey), (x, y), (0, 255, 0), 3, lineType=cv2.LINE_AA)
            cv2.circle(im, (x, y), 5, (0, 255, 0), -1)
        if (i > 12) and (i <= 16):
            cv2.line(im, (prex, prey), (x, y), (255, 255, 0), 3, lineType=cv2.LINE_AA)
            cv2.circle(im, (x, y), 5, (255, 255, 0), -1)
        if (i > 16) and (i <= 20):
            cv2.line(im, (prex, prey), (x, y), (255, 0, 0), 3, lineType=cv2.LINE_AA)
            cv2.circle(im, (x, y), 5, (255, 0, 0), -1)

        prex = x
        prey = y

    return im


def draw_landmarks(img, landmarks):
    num_joints = landmarks.shape[0]
    for i in range(num_joints):
        x, y = landmarks[i][:2]
        img = cv2.circle(img, (int(x), int(y)), radius=3, color=(0, 0, 255), thickness=-1)
    return img


def main(json_path, img_dir, mano_left_path, mano_right_path):
    lhand_reg, rhand_reg = _init_mano(mano_left_path, mano_right_path)

    data = load_dataset(json_path)

    num_imgs = len(data['images'])
    num_annos = len(data['annotations'])
    print(f'num_imgs: {num_imgs}')
    print(f'num_annos: {num_annos}')

    # print("Data keys:", [k for k in data.keys()])
    # print("Image keys:", [k for k in data['images'][0].keys()])
    # print("Annotations keys:", [k for k in data['annotations'][0].keys()])
    #
    # print("The number of images:", len(data['images']))
    # print("The number of annotations:", len(data['annotations']))

    num_counts = 0
    num_not_find_imgs = 0

    for i in range(num_imgs):
        img_info = data['images'][i]
        h, w = img_info["height"], img_info["width"]
        img_path = os.path.join(img_dir, img_info["name"].replace('.png', '.jpg'))
        img_id = img_info["id"]

        anno_infos = list()
        for j in range(num_annos):
            anno_info = data['annotations'][j]
            if anno_info['image_id'] == img_id:
                anno_infos.append(anno_info)

        if os.path.exists(img_path):
            img = cv2.resize(cv2.imread(img_path), (w, h), interpolation=cv2.INTER_CUBIC)

            # print(f'h: {h}, w: {w}')
            # print(f'img_path: {img_path}')
            # print(f'img_id: {img_id}')
            for anno_info in anno_infos:
                # print(f"is_left: {anno_info['is_left']}")
                vertics = np.array(anno_info['vertices'])
                if int(anno_info['is_left']) == 0:
                    landmarks = np.matmul(rhand_reg, vertics)
                else:
                    landmarks = np.matmul(lhand_reg, vertics)

                # add tips
                finger_tips = np.zeros((len(FINGERTIP_IDXS_MANO), 3), dtype=float)
                for k, index_finger in enumerate(FINGERTIP_IDXS_MANO):
                    finger_tips[k] = vertics[index_finger].copy()

                new_landmarks = landmarks_mappling(landmarks, finger_tips)
                show_img = draw_point(new_landmarks, img)

                num_counts += 1
        else:
            num_not_find_imgs += 1

        cv2.imshow("Show", show_img)
        if cv2.waitKey(0) == ord('q'):
            exit("Exit!")

    print(f'num_counts: {num_counts}')
    print(f'num_not_find_imgs: {num_not_find_imgs}')


if __name__ == "__main__":
    json_path = r"D:\Data\landmarks\YouTube-3D-Hands\youtube_test.json"
    img_dir = r"H:\data\data"
    mano_left_path = r"F:\BOBBY\Code\H0100_mano_v1_2\models\MANO_LEFT.pkl"
    mano_right_path = r"F:\BOBBY\Code\H0100_mano_v1_2\models\MANO_RIGHT.pkl"

    main(json_path, img_dir, mano_left_path, mano_left_path)
