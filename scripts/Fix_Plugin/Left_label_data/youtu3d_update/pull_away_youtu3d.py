import copy
import json
import os.path

from tqdm import tqdm
from json_tools import make_json_head


mode = 'train'

json_dir = fr'E:\left_hand_label_data\annotations\person_keypoints_{mode}2017.json'
save_path = r'E:\left_hand_label_data\annotations\youtu3d_update'

def main():
    json_head = make_json_head()
    youtu_head = copy.deepcopy(json_head)
    others_head = copy.deepcopy(json_head)
    with open(json_dir, 'r')as f:
        json_data = json.load(f)
        images = json_data['images']
        annotations = json_data['annotations']

    iter_num = len(images)
    for i in tqdm(range(iter_num)):
        image_info = images[i]
        annotation_info = annotations[i]

        image_id = annotation_info['image_id']
        if image_id >= 400_000:
            others_head['images'].append(image_info)
            others_head['annotations'].append(annotation_info)

        else:
            youtu_head['images'].append(image_info)
            youtu_head['annotations'].append(annotation_info)

    save_dir1 = os.path.join(save_path, f"youtu_{mode}.json")
    with open(save_dir1, 'w')as f:
        json.dump(youtu_head, f)

    save_dir2 = os.path.join(save_path, f"others_{mode}.json")
    with open(save_dir2, 'w')as f:
        json.dump(others_head, f)



if __name__ == '__main__':
    main()