"""
对于这个10min新的json文件，我们开始采用一张图片一个json单元存储；而COCO-json格式则是一个手一个json单元，该程序就是转成coco-json格式，方便后面跑coco_eval函数
"""
import json
import numpy as np
from tqdm import tqdm
from copy import deepcopy

from library.json_tools import make_json_head
from convert_tools import convert_coco_format_from_wholebody


def main():
    coco_id = 1_400_000
    mode = 'test2017'
    json_head = make_json_head()
    json_dir = r'E:\test_data\test_data_from_whole_body\annotations\average_pseudo_labels.json'
    save_dir = r'E:\test_data\test_data_from_whole_body\annotations\average_pseudo_labels-coco.json'
    with open(json_dir, 'r')as f:
        json_data = json.load(f)
        images = json_data['images']
        annotations = json_data['annotations']
        assert (len(images) == len(annotations))

    iter_num = len(images)
    for i in tqdm(range(iter_num)):
        image_info = images[i]
        annotation_info = annotations[i]
        image_dir = image_info['image_dir']
        keypoints_list = annotation_info['keypoints']

        for keypoints in keypoints_list:
            landmarks = np.array(keypoints).reshape(21, 3)
            if np.all(landmarks == 0):
                continue
            flag = convert_coco_format_from_wholebody(image_dir, landmarks, json_head, coco_id)
            if flag:
                coco_id += 1
            # coco_kps = np.array(keypoints).reshape(21, 3)
            # coco_kps[:, 2] = 2
            # coco_kps = coco_kps.flatten().tolist()
            # new_annotation_info['keypoints'] = coco_kps
            # json_head['images'].append(new_image_info)
            # json_head['annotations'].append(new_annotation_info)

    print(f"writing the >>{save_dir}<< ......")
    with open(save_dir, 'w')as f:
        json.dump(json_head, f)
        print("Success to write in json!")


if __name__ == '__main__':
    main()
