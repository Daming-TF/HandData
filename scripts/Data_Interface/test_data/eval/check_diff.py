import json
import os.path
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import cv2
from copy import deepcopy

from library.tools import draw_2d_points, VideoWriter


def load_json_data(json_dir):
    with open(json_dir, 'r')as f:
        json_data = json.load(f)
        images = json_data['images']
        annotations = json_data['annotations']

    annotations_dict, images_dict = defaultdict(list), defaultdict(list)

    iter_num = len(images)
    for i in tqdm(range(iter_num)):
        image_info = images[i]
        annotation_info = annotations[i]
        assert (image_info['id'] == annotation_info['image_id'])

        image_id = annotation_info['image_id']
        images_dict[f'{image_id}'].append(image_info)
        annotations_dict[f'{image_id}'].append(annotation_info)

    return images_dict, annotations_dict

def get_image(image, image_id, images_dict, annotations_dict, txt=''):
    if str(image_id) in images_dict.keys():
        annotation_info_list = annotations_dict[str(image_id)]
        for annotation_info in annotation_info_list:
            keypoints = annotation_info['keypoints']
            image = draw_2d_points(np.array(keypoints).reshape(21, 3), image)
    image = cv2.putText(image, txt, (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    image = cv2.resize(image, (0, 0), fx=0.7, fy=0.7, interpolation=cv2.INTER_CUBIC)
    return image


def main():
    # cv2.namedWindow("test", cv2.WINDOW_NORMAL)
    # video_name = ["hand_test_04", "hand_test_05", "hand_test_06",
    #               "hand_test_07", "hand_test_08", "hand_test_09", "hand_test_10"]
    video_name = ["hand_test_01",]
    image_path = r'E:\test_data\test_data_from_whole_body\images'

    for video_id in video_name:
        gt_dir = fr'E:\test_data\test_data_from_whole_body\annotations\coco_eval\gt\{video_id}\{video_id}--gt.json'
        dt_v3_base_dir = fr'E:\test_data\test_data_from_whole_body\annotations\coco_eval\dt\{video_id}\{video_id}_v3-base.json'
        dt_v3_align_dir = fr'E:\test_data\test_data_from_whole_body\annotations\coco_eval\dt\{video_id}\{video_id}_v3-align.json'
        dt_v2_full_dir = fr'E:\test_data\test_data_from_whole_body\annotations\coco_eval\dt\{video_id}\{video_id}_v2-full.json'
        dt_v1_dir = fr'E:\test_data\test_data_from_whole_body\annotations\coco_eval\dt\{video_id}\{video_id}_v1.json'
        dt_mediapipe_full_dir = fr'E:\test_data\test_data_from_whole_body\annotations\coco_eval\dt\{video_id}\{video_id}--mediapipe.json'

        video_name = fr'{video_id}.mp4'
        video_dir = fr'E:\test_data\test_video\{video_name}'
        save_dir = fr'E:\test_data\test_data_from_whole_body\annotations\coco_eval\com-{video_name}'

        print(f"loading the '{gt_dir}'......")
        gt_images_dict, gt_annotations_dict = load_json_data(gt_dir)
        print(f"loading the '{dt_v3_base_dir}'......")
        dt_v3_base_images_dict, dt_v3_base_annotations_dict = load_json_data(dt_v3_base_dir)
        # print(f"loading the '{dt_v3_align_dir}'......")
        # dt_v3_align_images_dict, dt_v3_align_annotations_dict = load_json_data(dt_v3_align_dir)
        # print(f"loading the '{dt_v2_full_dir}'......")
        # dt_v2_full_images_dict, dt_v2_full_annotations_dict = load_json_data(dt_v2_full_dir)
        # print(f"loading the '{dt_v1_dir}'......")
        # dt_v1_images_dict, dt_v1_annotations_dict = load_json_data(dt_v1_dir)
        # print(f"loading the '{dt_mediapipe_full_dir}'......")
        # dt_mediapipe_full_images_dict, dt_mediapipe_full_annotations_dict = load_json_data(dt_mediapipe_full_dir)

        image_id_list = list(gt_images_dict.keys())

        cap = cv2.VideoCapture(video_dir)
        w, h = (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)*2.1),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        writer = VideoWriter(save_dir, cap, (w, h))

        for image_id in image_id_list:
            gt_image_info_list = gt_images_dict[str(image_id)]
            gt_annotation_info_list = gt_annotations_dict[str(image_id)]

            image_dir = os.path.join(image_path, gt_image_info_list[0]['file_name'])
            image = cv2.imread(image_dir)
            gt_image = deepcopy(image)

            for gt_annotation_info, gt_image_info in zip(gt_annotation_info_list, gt_image_info_list):
                keypoints = gt_annotation_info['keypoints']
                gt_image = draw_2d_points(np.array(keypoints).reshape(21, 3), gt_image)
            gt_image = cv2.putText(gt_image, "gt", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            gt_image = cv2.resize(gt_image, (0, 0), fx=0.7, fy=0.7, interpolation=cv2.INTER_CUBIC)

            dt_v3_base_image = get_image(deepcopy(image), image_id,
                                    dt_v3_base_images_dict, dt_v3_base_annotations_dict, 'v3')
            # dt_v3_align_image = get_image(deepcopy(image), image_id,
            #                             dt_v3_align_images_dict, dt_v3_align_annotations_dict, 'v3-align')
            # dt_v2_full_image = get_image(deepcopy(image), image_id,
            #                         dt_v2_full_images_dict, dt_v2_full_annotations_dict, 'v2-full')
            # dt_v1_image = get_image(deepcopy(image), image_id,
            #                     dt_v1_images_dict, dt_v1_annotations_dict, 'v1')
            # dt_mediapipe_full_image = get_image(deepcopy(image), image_id,
            #         dt_mediapipe_full_images_dict, dt_mediapipe_full_annotations_dict, 'mediapipe-full')

            # dt_v3_base_image = deepcopy(image)
            # if str(image_id) in dt_v3_base_images_dict.keys():
            #     dt_v3_base_annotation_info_list = dt_v3_base_annotations_dict[str(image_id)]
            #     for dt_v3_base_annotation_info in dt_v3_base_annotation_info_list:
            #         keypoints = dt_v3_base_annotation_info['keypoints']
            #         dt_v3_base_image = draw_2d_points(np.array(keypoints).reshape(21, 3), dt_v3_base_image)
            # dt_v3_base_image = cv2.putText(dt_v3_base_image, "v2-full", (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 2)

            canves = np.concatenate([dt_v3_base_image, gt_image], axis=1)

            print(image_dir)
            cv2.imshow("test", canves)
            cv2.waitKey(0)
            # writer.write(canves)

        writer.release()
        cap.release()


if __name__ == '__main__':
    main()
