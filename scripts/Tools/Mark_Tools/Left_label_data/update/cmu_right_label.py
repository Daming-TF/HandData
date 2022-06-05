import json
from tqdm import tqdm

from library.json_tools import make_json_head


json_dir = r'E:\left_hand_label_data\annotations\person_keypoints_train2017.json'
save_dir = r'E:\left_hand_label_data\annotations\cmu_update\person_keypoints_train2017.json'


def main():
    json_head = make_json_head()
    with open(json_dir, 'r')as f:
        json_data = json.load(f)
        images = json_data['images']
        annotations = json_data['annotations']

        iter_num = len(images)

    for i in tqdm(range(iter_num)):
        image_info = images[i]
        annotation_info = annotations[i]

        hand_type = annotation_info['hand_type']
        image_id = annotation_info['image_id']

        if 1_000_000 > image_id >= 900_000:
            if hand_type == 'left':
                annotation_info['hand_type'] = 'right'

        json_head['images'].append(image_info)
        json_head['annotations'].append(annotation_info)

    with open(save_dir, 'w')as f:
        json.dump(json_head, f)


if __name__ == '__main__':
    main()
