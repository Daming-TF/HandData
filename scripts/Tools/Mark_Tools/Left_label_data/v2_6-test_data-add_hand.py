'''
    通过旋转操作调正然后输入到封装的Mediapipe接口，得到左右手label
'''
import os
import cv2
from json_tools import make_json_head
import numpy as np
import json
from tqdm import tqdm
from rotation_tools import HandInfo, HandLandModel
from convert_coco_format import convert_coco_format_v2_6
from copy import deepcopy
from json_tools import load_json_data
from tools import draw_2d_points


def get_box(keypoints, box_factor=1):
    box = []
    hand_min = np.min(keypoints, axis=0)
    hand_max = np.max(keypoints, axis=0)
    hand_box_c = (hand_max + hand_min) / 2.
    half_size = int(np.max(hand_max - hand_min) * box_factor / 2.)

    x_left = int(hand_box_c[0] - half_size)
    y_top = int(hand_box_c[1] - half_size)
    x_right = x_left + 2 * half_size
    y_bottom = y_top + 2 * half_size
    return box[(x_left, y_top), (x_right, y_bottom)]


def rotation(image_dir, keypoints, alignmenter):
    ori_joints = np.array(keypoints).reshape(21, 3)
    data_numpy = cv2.imread(image_dir)
    # data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)
    input_data, warp_matrix = alignmenter(data_numpy, ori_joints)
    return input_data


def init(txt_path, save_path, debug):
    open(txt_path, 'w').close()
    if debug:
        cv2.namedWindow("aa", cv2.WINDOW_NORMAL)
    is_exists = os.path.exists(save_path)
    if not is_exists:
        os.makedirs(save_path)
        print('path of %s is build' % save_path)

    json_file = make_json_head()

    hand_mode = HandLandModel(capability=1)

    image_size = [224, 224]
    alignmenter = HandInfo(img_size=image_size[0])
    return json_file, hand_mode, alignmenter

def main():
    mode = 'test'
    json_path = fr'E:\test_data\test_data_from_whole_body\annotations\average_pseudo_labels_update-coco_id.json'
    txt_path = fr"E:\left_hand_label_data\annotations\v2_6\test_data_add_hand\mediapipe-detecter\record.txt"
    save_path = r"E:\left_hand_label_data\annotations\v2_6\test_data_add_hand\mediapipe-detecter"
    json_save_name = f'{mode}.json'
    badcase_save_name = f'badcase_{mode}.json'

    debug = 0

    json_head, hand_mode, alignmenter = init(txt_path, save_path, debug)
    goodcase_json = deepcopy(json_head)
    badcase_json = deepcopy(json_head)

    images_dict, annotations_dict = load_json_data(json_path)
    image_ids = list(images_dict.keys())

    success_count = 0
    badcase_count = 0
    for index, image_id in enumerate(image_ids):
        print(f"image_id:{image_id}\tremain:{len(image_ids)-index}\tsuccess:{success_count}\tbadcase:{badcase_count}")
        image_unit = images_dict[image_id]
        anno_unit_list = annotations_dict[image_id]

        image_dir = image_unit['image_dir']
        image = cv2.imread(image_dir)

        image_dir = image_unit['image_dir']
        for anno_unit in anno_unit_list:
            keypoints = anno_unit['keypoints']
            bbox = anno_unit['bbox']

            input_data = rotation(image_dir, keypoints, alignmenter)
            landmark, handness, righthand_prop, _ = hand_mode.run(input_data)

            if righthand_prop > 0.5:
                hand_type = 'right'
            else:
                hand_type = 'left'

            if debug:
                img = cv2.putText(input_data, f'{hand_type}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1)
                # img = draw_2d_points(landmark, deepcopy(img))
                cv2.imshow('aa', img)
                cv2.waitKey(0) & 0xFF

            if handness < 0.5:
                with open(txt_path, 'a') as f:
                    f.write(image_dir + '\n')
                    print(f'No hands\t{image_dir}')
                convert_coco_format_v2_6(badcase_json, image_unit, anno_unit,
                                               hand_type=None, handness=handness[0])
                badcase_count += 1

            else:
                convert_coco_format_v2_6(goodcase_json, image_unit, anno_unit,
                                               hand_type=hand_type, handness=handness[0])
                success_count += 1

    json_path = os.path.join(save_path, json_save_name)
    with open(json_path, 'w') as fw:
        json.dump(goodcase_json, fw)
        print(f"{json_path} have succeed to write")

    json_path = os.path.join(save_path, badcase_save_name)
    with open(json_path, 'w') as fw:
        json.dump(badcase_json, fw)
        print(f"{json_path} have succeed to write")


if __name__ == "__main__":
    main()