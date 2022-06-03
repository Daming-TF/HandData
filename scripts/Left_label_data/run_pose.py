'''
    通过肢体姿态检测，得到左右手label
'''
import os
import cv2
from json_tools import make_json_head
from pose_tool import PoseLandmark, landmark_to_box, bb_iou
import numpy as np
import json
from tqdm import tqdm
from convert_coco_format import convert_coco_format_left_label
from copy import deepcopy

mode = 'val'
# E:\whole_body_data\annotations\person_keypoints_{mode}2017.json
json_dir = fr'E:\left_hand_label_data\detect\{mode}_badcase.json'
txt_dir = fr"E:\left_hand_label_data\pose\record_{mode}.txt"
SAVE_PATH = r"E:\left_hand_label_data\pose"
JSON_NAME = f'{mode}.json'
debug = 1


def init():
    if debug:
        cv2.namedWindow("aa", cv2.WINDOW_NORMAL)
    is_exists = os.path.exists(SAVE_PATH)
    if not is_exists:
        os.makedirs(SAVE_PATH)
        print('path of %s is build' % SAVE_PATH)

    json_file = make_json_head()
    hand_mode = PoseLandmark()
    return json_file, hand_mode


def main():
    json_head, hand_mode = init()

    with open(json_dir, 'r')as f:
        json_data = json.load(f)
        images = json_data['images']
        annotations = json_data['annotations']
        assert (len(images) == len(annotations))

    count = 0
    for index in tqdm(range(len(images))):
        # if index < 1500:
        #     continue
        image_info = images[index]
        annotation_info = annotations[index]

        image_dir = image_info['image_dir']
        keypoints = annotation_info['keypoints']

        image = cv2.imread(image_dir)
        json_landmarks = np.array(keypoints).reshape(21, 3)
        json_box = landmark_to_box(json_landmarks)
        boxes, _ = hand_mode(image)

        iou_list = []
        for box in boxes:
            pose_box = box['box']
            bb_iou(json_box, pose_box)
            iou_list.append(bb_iou)

        index = max(iou_list)
        hand_type = boxes[index]['type']

        if debug:
            img = cv2.putText(image, f'{hand_type}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1)
            cv2.imshow('aa', img)
            cv2.waitKey(0) & 0xFF

        convert_coco_format_left_label(json_head, deepcopy(image_info), deepcopy(annotation_info), hand_type,
                                       hand_type)

    json_path = os.path.join(SAVE_PATH, JSON_NAME)
    with open(json_path, 'w') as fw:
        json.dump(json_head, fw)
        print(f"{json_path} have succeed to write")





if __name__ == '__main__':
    main()