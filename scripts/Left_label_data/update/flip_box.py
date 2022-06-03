import json

import cv2
from tqdm import tqdm
from json_tools import make_json_head

json_dir = r'G:\imgdate2\HO3D_v3\HO3D_from_whole_body\annotations\person_keypoints_train2017-update.json'
save_dir = r'G:\imgdate2\HO3D_v3\HO3D_from_whole_body\annotations\person_keypoints_train2017-flip_update.json'

def main():
    json_head = make_json_head()
    with open(json_dir, 'r')as f:
        json_data = json.load(f)
        images = json_data['images']
        annotations = json_data['annotations']

    iter_num = len(images)

    for i in tqdm(range(iter_num)):
        images_info = images[i]
        annotation_info = annotations[i]

        image_id = annotation_info['image_id']
        if 1_300_000 <= image_id < 1_339_806:
            width = images_info['width']
            bbox = annotation_info['bbox']
            x_left, y_top, h, w = bbox
            x_left = width-(x_left + w/2)-w/2
            bbox[0] = int(x_left)

            # bbox[1] = int(height) - bbox[1]

            annotation_info['bbox'] = bbox

        json_head['images'].append(images_info)
        json_head['annotations'].append(annotation_info)

    with open(save_dir, 'w')as f:
        json.dump(json_head, f)


if __name__ == '__main__':
    main()