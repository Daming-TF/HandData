import json
import os.path
from tqdm import tqdm
import numpy as np

from convert_tools import convert_coco_format_from_wholebody
from library.json_tools import make_json_head


def main():
    mode ='dt'
    gt_suffix = '_update.json'
    dt_suffix = '-Mediapipe_Full.json'
    vedio_names = ["hand_test_01", "hand_test_02", "hand_test_03", "hand_test_04", "hand_test_05", "hand_test_06",
                   "hand_test_07", "hand_test_08", "hand_test_09", "hand_test_10"]
    json_path = fr'E:\test_data\test_data_from_whole_body\annotations\coco_eval\{mode}'
    # save_path = fr'E:\test_data\test_data_from_whole_body\annotations\coco_eval\{mode}'

    for vedio_name in vedio_names:
        print(f"Converting to {vedio_name}")
        json_head = make_json_head()
        if mode == 'gt':
            json_dir = os.path.join(json_path, vedio_name, vedio_name+gt_suffix)
        if mode == 'dt':
            json_dir = os.path.join(json_path, vedio_name, vedio_name+dt_suffix)

        with open(json_dir, 'r')as f:
            json_data = json.load(f)
            images = json_data["images"]
            annotations = json_data["annotations"]

            iter_num = len(images)

            for i in tqdm(range(iter_num)):
                image_info = images[i]
                annotation_info = annotations[i]

                file_name = image_info["file_name"]
                image_dir = image_info["image_dir"]
                hands_keypoints = annotation_info["keypoints"]
                image_id = annotation_info["image_id"]

                for index, keypoints in enumerate(hands_keypoints):
                    keypoints = np.array(keypoints).reshape(21, 3)
                    if np.all(keypoints == 0):
                        continue
                    convert_coco_format_from_wholebody(json_head, image_dir, image_id, file_name, keypoints)

        save_path, save_name = os.path.split(json_dir)
        save_name = save_name.split('.')[0]+'-coco.json'
        save_dir = os.path.join(save_path, save_name)
        with open(save_dir, 'w')as f:
            json.dump(json_head, f)
            print(f"success to write in {save_dir}")


if __name__ == "__main__":
    main()
