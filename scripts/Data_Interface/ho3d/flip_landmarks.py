import json

import cv2
from tqdm import tqdm
from json_tools import make_json_head
from tools import draw_2d_points
import numpy as np
from copy import deepcopy

mode = 'train'
json_dir = fr'G:\imgdate2\HO3D_v3\HO3D_from_whole_body\annotations\person_keypoints_{mode}2017.json'
save_dir = fr'G:\imgdate2\HO3D_v3\HO3D_from_whole_body\annotations\person_keypoints_{mode}2017_flip.json'

def main():
    json_head = make_json_head()
    with open(json_dir)as f:
        json_data = json.load(f)
        images = json_data['images']
        annotations = json_data['annotations']
        assert(len(images) == len(annotations))
    for i in tqdm(range(len(annotations))):
        image_info = images[i]
        annotation_info = annotations[i]
        width = image_info['width']
        image_dir = image_info['image_dir']
        keypoints = annotation_info['keypoints']
        prieds = np.array(keypoints).reshape(21, 3)

        update_prieds = np.zeros([21, 3])
        update_prieds[:, 0] = width - prieds[:, 0]
        update_prieds[:, 1:3] = prieds[:, 1:3]

        img = cv2.imread(image_dir)
        img = draw_2d_points(update_prieds, img)
        cv2.imshow('a', img)
        cv2.waitKey(1)

        new_image_info = deepcopy(image_info)
        new_annotation_info = deepcopy(annotation_info)
        new_annotation_info['keypoints'] = update_prieds.flatten().tolist()
        json_head['images'].append(new_image_info)
        json_head['annotations'].append(new_annotation_info)

    with open(save_dir, 'w')as f:
        json.dump(json_head, f)










if __name__ == '__main__':
    main()
