import json
import os

from tqdm import tqdm
from json_tools import make_json_head

save_path = r'E:\test_data\test_data_from_whole_body\annotations\coco_eval\gt'
vedio_name = ["hand_test_01", "hand_test_02", "hand_test_03", "hand_test_04", "hand_test_05", "hand_test_06",
              "hand_test_07", "hand_test_08", "hand_test_09", "hand_test_10"]
start_id_list = [1_402_094, 1_404_081, 1_404_682, 1_410_139, 1_410_905, 1_412_448,
            1_414_397, 1_416_186, 1_419_204, 1_423_669]

# mediapipe_full-vedio-coco_id.json, mediapipe_lite-vedio-coco_id.json, average_pseudo_labels_update-coco_id.json,
json_dir = r"E:\test_data\test_data_from_whole_body\annotations\average_pseudo_labels_update-coco_id.json"
suffix = '-gt.json'        # --mediapipe.json, --mediapipe-lite.json, --gt.json

def main():
    with open(json_dir, 'r')as f:
        json_data = json.load(f)
        images = json_data['images']
        annotations = json_data['annotations']

    json_head = make_json_head()
    iter_num = len(images)

    j = 0
    for i in tqdm(range(iter_num)):
        image_info = images[i]
        annotation_info = annotations[i]
        annotation_info['score'] = 1

        if image_info['id'] > start_id_list[j]:
            save_name = vedio_name[j]
            save_dir = os.path.join(save_path, save_name)
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)

            print(f"Writing in {save_dir}")
            save_dir = os.path.join(save_dir, save_name+suffix)
            with open(save_dir, 'w')as f:
                json.dump(json_head, f)
            print(f"There are {len(json_head['images'])} data!")

            j += 1

            json_head = make_json_head()
            assert (len(json_head['images']) == len(json_head['annotations']) == 0)
            json_head['images'].append(image_info)
            json_head['annotations'].append(annotation_info)
            continue

        json_head['images'].append(image_info)
        json_head['annotations'].append(annotation_info)

    save_name = vedio_name[j]
    save_dir = os.path.join(save_path, save_name)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    print(f"Writing in {save_dir}")
    save_dir = os.path.join(save_dir, save_name+suffix)
    with open(save_dir, 'w') as f:
        json.dump(json_head, f)
    print(f"There are {len(json_head['images'])} data!")


if __name__ == "__main__":
    main()