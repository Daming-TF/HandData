import json
import cv2
from rotation_tools import HandInfo
import numpy as np
from tqdm import tqdm

json_dir = r'E:\Data\landmarks\handpose_x_gesture_v1\HXG_from_whole_body\annotations\person_keypoints_val2017-update-for-invaliddata.json'
image_size = [224, 224]

def main():
    with open(json_dir, 'r')as f:
        json_date = json.load(f)
        images = json_date['images']
        annotations = json_date['annotations']
    assert (len(json_date['images']) == len(json_date['annotations']))

    for i in tqdm(range(len(images))):
        image_info = images[i]
        annotation_info = annotations[i]

        image_dir = image_info['image_dir']
        keypoints = annotation_info['keypoints']

        ori_joints = np.array(keypoints).reshape(21, 3)
        data_numpy = cv2.imread(
            image_dir, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
        )
        data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)
        alignmenter = HandInfo(img_size=image_size[0])
        input_data, warp_matrix = alignmenter(data_numpy, ori_joints)
        cv2.imshow('a', input_data)
        cv2.waitKey(0)




if __name__ == '__main__':
    main()