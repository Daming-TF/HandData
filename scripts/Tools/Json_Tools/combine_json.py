import os
import sys
import json
from tqdm import tqdm

pro_path = os.path.dirname(os.path.dirname(__file__))
sys.path.append(pro_path)

from library.json_tools import make_json_head

json_num = 5


def main(save_name, json_list):
    json_file = make_json_head()  # init new json file

    for i in range(len(json_list)):
        json_path = json_list[i]
        print(f"combining {json_path}.......")
        combine_json(json_file, json_path)

    # Dump to the new json file
    with open(save_name, 'w') as fw:
        json.dump(json_file, fw)
    print(f'SUCCESS')


def combine_json(json_file, json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    # check number of items
    num_imgs = len(data['images'])
    num_annos = len(data['annotations'])
    # assert num_imgs == num_annos
    print(f'{os.path.basename(json_path)} load SUCCESS!')

    for i in tqdm(range(num_imgs)):
        if data['images'][i]['id'] == 1400026:
            print()
        json_file['images'].append(data['images'][i])
    print(f'Finish to write json --{os.path.basename(json_path)}')

    for i in tqdm(range(num_annos)):
        json_file['annotations'].append(data['annotations'][i])
    print(f'Finish to write json --{os.path.basename(json_path)}')

    return json_file


if __name__ == "__main__":
    mode = 'val'
    # path = r'E:\left_hand_label_data\annotations\youtu3d_update'
    json_name = fr'person_keypoints_{mode}2017_update.json'

    json_path_1 = \
        fr"E:\Data\landmarks\YouTube3D\YouTube3D_from_whole_body_v2_6\annotations\{json_name}"
    json_path_2 = \
        fr"E:\Data\landmarks\HFB\HFB_from_whole_body_v2_6\annotations\{json_name}"
    json_path_3 = \
        fr"E:\Data\landmarks\handpose_x_gesture_v1\HXG_from_whole_body_v2_6\annotations\{json_name}"
    json_path_4 = \
        fr"E:\Data\landmarks\FH\FH_from_whole_body_v2_6\annotations\{json_name}"
    json_path_5 = \
        fr"F:\image\CMU\hand_labels\hand_labels_from_whole_body_v2_6\annotations\{json_name}"
    json_path_6 = \
        fr"F:\image\CMU\hand143_panopticdb\hand143_panopticdb_from_whole_body_v2_6\annotations\{json_name}"
    json_path_7 = \
        fr"F:\image\CMU\hand_labels_synth\hand_labels_synth_from_whole_body_v2_6\annotations\{json_name}"
    json_path_8 = \
        fr"F:\image\Rendered Handpose Dataset Dataset\RHD\RHD_from_whole_body_v2_6\annotations\{json_name}"
    json_path_9 = \
        fr"F:\image\COCO_whole_body\coco_from_whole_body_v2_6\annotations\{json_name}"
    json_path_10 = \
        fr"G:\imgdate2\HO3D_v3\HO3D_from_whole_body_v2_6\annotations\{json_name}"
    json_path_11 = \
        fr"G:\test_data\new_data\new_data_from_whole_body_v2_6\annotations\{json_name}"
    json_path_12 = \
        fr"G:\test_data\hardcase_data\hardcase_from_whole_body_v2_6\annotations\{json_name}"

    #   ---------------------------------------------------------------------------------------------------
    # json_path_1 = \
    #     fr"{path}/{mode}2017_update.json"  # fr"{path}\2022-3-21-18-35-46.json"
    # json_path_2 = \
    #     fr"{path}/others_{mode}.json"

    # # person_keypoints_train2017.json
    save_name_ = \
        fr"G:\transmission\anno\person_keypoints_{mode}2017.json"

    json_num = 12

    json_list = []
    for i in range(1, json_num+1):
        exec("json_dir = json_path_{}".format(i))
        json_path = locals()["json_dir"]
        if os.path.exists(json_path):
            json_list.append(json_path)

    main(save_name_, json_list)
