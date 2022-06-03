import json
import numpy as np

from tqdm import tqdm

json_dir = r'E:\left_hand_label_data\annotations\person_keypoints_train2017.json'


def main():
    with open(json_dir, 'r')as f:
        json_data = json.load(f)
        images = json_data['images']
        annotations = json_data['annotations']

        iter_num = len(images)

    for i in range(iter_num):
        image_info = images[i]
        annotation_info = annotations[i]

        keypoints = annotation_info['keypoints']
        image_id = annotation_info['image_id']
        # if int(image_id) >= 400_000:
        #     continue

        keypoints = np.array(keypoints).reshape(21, 3)
        kps_valid_bool = keypoints[:, -1].astype(bool)
        key_pts = keypoints[:, :2][kps_valid_bool]

        average_point = np.mean(key_pts)
        l2_max = 0

        for j in range(key_pts.shape[0]):
            l2 = np.sqrt(sum(np.power((average_point-key_pts[j]), 2)))
            if l2 > l2_max:
                l2_max = l2
        if l2_max > 1000:
            print(f"image id:{image_id}\tl2:{l2_max}")

if __name__ == '__main__':
    main()