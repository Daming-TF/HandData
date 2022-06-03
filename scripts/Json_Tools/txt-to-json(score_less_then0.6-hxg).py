'''
    通过遍历badcase数据路径来匹配对应的全图数据（搜索所有图片），当匹配上的时候把对应手关键点放入到容器中继续匹配剩余数据
'''
import json
import os

import numpy as np
from json_tools import _init_save_folder
from tqdm import tqdm
import re
import copy

# badcase_txt = r'E:\Data\landmarks\FreiHAND_pub_v2\badcase\total_badcase.txt'
data_json_path = r'E:\Data\landmarks\handpose_x_gesture_v1\handpose_x_gesture_v1\badcase\score_less_0.6.json'
save_path = r'E:\Data\landmarks\handpose_x_gesture_v1\handpose_x_gesture_v1\badcase\upload json2'
# data_name = 'cmu-real'       # 'ho3d'

def main():
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    with open(data_json_path, 'r')as f_json_data:
        json_data = json.load(f_json_data)
        image_data = json_data['images']
        anno_data = json_data['annotations']
        data_length = len(image_data)

    json_head_start = _init_save_folder()

    for i in range(data_length):
        keypoints = list()
        json_head = copy.deepcopy(json_head_start)
        image_info = image_data[i]
        annotation_info = anno_data[i]
        image_dir = image_info['image_dir']

        a = list(map(int, [substr.start() for substr in re.finditer(r'\\', image_dir)]))
        mode = image_dir[a[-2] + 1: a[-1]]
        mode = mode.replace('\\', '_')

        prieds = annotation_info['keypoints']
        prieds = np.array(prieds).reshape(21, 3)[:, 0:2]
        keypoints.append(prieds.flatten().tolist())
        record_dir = image_dir
        flag = 1

        if flag == 1:
            img_id = os.path.basename(record_dir).split('.')[0]
            image_dict = dict({
                'license': 1,
                'id': img_id,
                'file_name': record_dir,
                'mode': mode
            })

            value = 1
            anno_dict = dict({
                'num_keypoints': 21,
                'keypoints': list(keypoints),
                'category_id': 1,
                'hand_value': value,
                'id': img_id
            })

            json_head['images'].append(image_dict)
            json_head['annotations'].append(anno_dict)

            save_dir = os.path.join(save_path, mode+'_'+img_id+'.json')    # os.path.join(save_path, img_id+'.json')
            if os.path.exists(save_dir):
                print(f"{save_dir} is exists")
            # 写入json文件
            with open(save_dir, 'w') as fw:
                json.dump(json_head, fw)

            del json_head


if __name__ == '__main__':
    main()