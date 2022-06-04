import json
from tqdm import tqdm
from convert_tools import convert_coco_format_from_wholebody
import numpy as np
from library.json_tools import make_json_head


def main():
    json_dir = r'E:\test_data\test_data_from_whole_body\annotations\mediapipe_lite-vedio.json'
    save_dir = r'E:\test_data\test_data_from_whole_body\annotations\mediapipe_lite-vedio-coco.json'
    json_head = make_json_head()
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

    with open(save_dir, 'w')as f:
        json.dump(json_head, f)
        print(f"success to write in {save_dir}")


if __name__ == "__main__":
    main()
