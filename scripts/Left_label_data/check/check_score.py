import json

import numpy as np
from tqdm import tqdm

json_dir = r'E:\left_hand_label_data\annotations\person_keypoints_val2017.json'

with open(json_dir, 'r')as f:
    json_data = json.load(f)
    annotations = json_data['annotations']

    iter_num = len(annotations)

    for i in tqdm(range(iter_num)):
        annotation_info = annotations[i]
        keypoints = np.array(annotation_info['keypoints']).reshape(21, 3)
        k_vaild = keypoints[:, -1]
        for i in range(k_vaild.shape[0]):
            k_vaild[i]
