import json
import os
import numpy as np
from collections import defaultdict
from tqdm import tqdm

from json_tools import make_json_head


def main():
    # cv2.namedWindow("test", cv2.WINDOW_NORMAL)
    whole_json_path = \
        r'G:\test_data\new_data\new_data_from_whole_body\annotations\test_update-update_bool.json'
    crop_json_path = \
        r'G:\test_data\new_data\new_data_from_whole_body\annotations_need_to_match\v2_4_person_keypoints_test2017.json'

    match_txt_path = r'G:\test_data\new_data\new_data_from_whole_body\match-v2_4-v2_5\match.txt'
    no_img_match_txt_path = r'G:\test_data\new_data\new_data_from_whole_body\match-v2_4-v2_5\img_no_match.txt'
    no_kps_bool_match_txt_path = r'G:\test_data\new_data\new_data_from_whole_body\match-v2_4-v2_5\bool_no_match.txt'

    whole_data_dir = r'G:\test_data\new_data\new_data_from_whole_body\new_images\test2017'
    crop_data_dir = r'G:\test_data\new_data\crop_images'

    save_path= r'G:\test_data\new_data\new_data_from_whole_body\annotations\test_update-update_bool.json'

    for txt_path in [match_txt_path, no_img_match_txt_path, no_kps_bool_match_txt_path]:
        open(txt_path, 'w').close()

    whole_images_dict, whole_annotations_dict = load_json_data(whole_json_path)
    crop_images_dict, crop_annotations_dict = load_json_data(crop_json_path)

    whole_ids = get_ids(whole_images_dict)
    crop_ids = get_ids(crop_images_dict)

    no_match_counter, match_counter = 0, 0
    for index in tqdm(range(len(crop_ids))):
        crop_image_id = crop_ids[index]
        crop_image_info = crop_images_dict[crop_image_id]
        crop_annotations_info = crop_annotations_dict[crop_image_id][0]

        crop_file_name = crop_image_info["file_name"]
        crop_keypoints = np.array(crop_annotations_info["keypoints"]).reshape(21, 3)
        crop_image_path = os.path.join(crop_data_dir, crop_file_name)

        find_flag = 0

        for i, whole_image_id in enumerate(whole_ids):
            whole_image_info = whole_images_dict[whole_image_id]
            whole_annotation_info_list = whole_annotations_dict[whole_image_id]

            for whole_annotation_info in whole_annotation_info_list:
                whole_file_name = whole_image_info["file_name"]
                whole_keypoints = np.array(whole_annotation_info["keypoints"]).reshape(21, 3)
                whole_image_path = os.path.join(whole_data_dir, whole_file_name)

                crop_keypoints_from_whole = np.zeros([21, 3])
                crop_keypoints_from_whole[:, 2] = whole_keypoints[:, 2]
                crop_keypoints_from_whole[:, :2] = crop_box_wo_img(whole_keypoints)

                # if crop_w == crop_image_from_whole.shape[0]:
                #     crop_image = cv2.imread(crop_image_path)
                #     diff = np.sum(np.abs(deepcopy(crop_image) - deepcopy(crop_image_from_whole)))/(crop_w*crop_w*3)
                #     cv2.imwrite(save_path, crop_image_from_whole)
                #     # print(f"file name:{crop_file_name}={file_name}\tdiff:{diff}")

                valid_bool_flage, kps_flage = get_match(crop_keypoints, crop_keypoints_from_whole)

                if kps_flage:
                    # print(f"file name:{crop_file_name}={whole_file_name}")
                    if valid_bool_flage is False:
                        update_keypoints(whole_annotation_info, crop_keypoints)
                        with open(no_kps_bool_match_txt_path, 'a') as f:
                            f.write(crop_image_path + '**' + whole_image_path + '\n')

                    with open(match_txt_path, 'a') as f:
                        f.write(crop_image_path+'**'+whole_image_path + '\n')
                    find_flag = 1

            if find_flag:
                match_counter += 1
                break

        if find_flag == 0:
            no_match_counter += 1
            with open(no_img_match_txt_path, 'a') as f:
                f.write(crop_image_path + '**' + whole_image_path + '\n')
            print(f"[!] no match between >>{crop_image_path}<< and >>{whole_image_path}<<")

    print(f"No match:{no_match_counter}\tMatch:{match_counter}")

    write_json(whole_images_dict, whole_annotations_dict, save_path)


def write_json(images_dict, annotations_dict, save_path):
    json_head = make_json_head()
    images_ids = get_ids(images_dict)
    for image_id in images_ids:
        image_info = images_dict[image_id]
        json_head["images"].append(image_info)

        annotation_info_list = annotations_dict[image_id]
        for annotation_info in annotation_info_list:
            json_head["annotations"].append(annotation_info)

    with open(save_path, 'w')as f:
        json.dump(json_head, f)


def update_keypoints(annotation_info, crop_keypoints):
    keypoints = np.array(annotation_info["keypoints"]).reshape(21, 3)
    keypoints[:, 2] = crop_keypoints[:, 2]
    annotation_info["keypoints"] = keypoints.flatten().tolist()


def load_json_data(json_path):
    annotations_dict = defaultdict(list)
    images_dict = {}
    with open(json_path, 'r')as f:
        dataset = json.load(f)

    for ann in dataset['annotations']:
        annotations_dict[ann['image_id']].append(ann)

    for img in dataset['images']:
        images_dict[img['id']] = img

    return images_dict, annotations_dict


def get_ids(data_dict):
    return list(data_dict.keys())


def crop_box_wo_img(hand_pts_2d, box_factor=2.2):
    coco_kps = hand_pts_2d.copy()

    kps_valid_bool = coco_kps[:, -1].astype(bool)
    new_hand_pts_2d = coco_kps[:, :2][kps_valid_bool]

    hand_min = np.min(new_hand_pts_2d, axis=0)
    hand_max = np.max(new_hand_pts_2d, axis=0)
    hand_box_c = (hand_max + hand_min) / 2.
    half_size = int(np.max(hand_max - hand_min) * box_factor / 2.)

    x_left = int(hand_box_c[0] - half_size)
    y_top = int(hand_box_c[1] - half_size)

    save_pts = hand_pts_2d[:, :2] - np.array([x_left, y_top])

    return save_pts


def get_match(kps1, kps2):
    valid_bool_flag, kps_flag = False, False
    kps1_valid_bool = kps1[:, -1]
    kps1 = kps1[:, :2].flatten()

    kps2_valid_bool = kps2[:, -1]
    kps2 = kps2[:, :2].flatten()

    if np.sum(np.abs(kps1_valid_bool-kps2_valid_bool)) == 0:
        valid_bool_flag = True
    if np.sum(np.abs(kps1-kps2)) == 0:
        kps_flag = True

    return valid_bool_flag, kps_flag


if __name__ == '__main__':
    main()