from copy import deepcopy
from tqdm import tqdm

from library.json_tools import load_json_data, write_json, get_ids

json_path = fr'E:\left_hand_label_data\annotations\v2_6\person_keypoints_train2017-update.json'

counter = 0
images_dict, annotations_dict = load_json_data(json_path)
assert get_ids(images_dict) == get_ids(annotations_dict)
image_ids = deepcopy(get_ids(annotations_dict))

for image_id in tqdm(image_ids):
    anno_unit_list = annotations_dict[image_id]
    for anno_unit in anno_unit_list:
        hand_type = anno_unit['hand_type']
        if hand_type is None:
            counter += 1
            break

print(f"No hand num：{counter}")



