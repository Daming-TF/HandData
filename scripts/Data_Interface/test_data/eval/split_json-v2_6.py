import json
from copy import deepcopy
import os

from tqdm import tqdm
from json_tools import make_json_head, load_json_data

save_dir = r'E:\test_data\test_data_from_whole_body\annotations\coco_eval\gt'
video_names = ["hand_test_01", "hand_test_02", "hand_test_03", "hand_test_04", "hand_test_05", "hand_test_06",
              "hand_test_07", "hand_test_08", "hand_test_09", "hand_test_10"]
start_id_list = [1_402_094, 1_404_081, 1_404_682, 1_410_139, 1_410_905, 1_412_448,
            1_414_397, 1_416_186, 1_419_204, 1_423_669]

# mediapipe_full-vedio-coco_id.json, mediapipe_lite-vedio-coco_id.json, average_pseudo_labels_update-coco_id.json,
json_dir = r"E:\left_hand_label_data\annotations\v2_6\test_data_add_hand\person_keypoints_test2017.json"
suffix = '--gt.json'        # --mediapipe.json, --mediapipe-lite.json, --gt.json

def main():
    images_dict, annotations_dict = load_json_data(json_dir)
    json_head = make_json_head()

    start_id = 0
    for index, end_id in enumerate(start_id_list):
        json_file = deepcopy(json_head)
        video_id = video_names[index]
        for image_id in list(images_dict.keys()):
            if end_id >= image_id > start_id:
                image_info = images_dict[image_id]
                json_file['images'].append(image_info)

                annotations_info_list = annotations_dict[image_id]
                for annotations_info in annotations_info_list:
                    json_file['annotations'].append(annotations_info)

        start_id = end_id

        save_path = os.path.join(save_dir, video_id, video_id+suffix)
        with open(save_path, 'w')as f:
            json.dump(json_file, f)


if __name__ == "__main__":
    main()