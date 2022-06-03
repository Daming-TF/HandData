# encoding:utf-8
import numpy as np
import requests
from aip import AipBodyAnalysis
import os
import json
from tqdm import tqdm
from json_tools import make_json_head
import time
import cv2


def image_data(data_dir):
    img_dirs = []
    filenames = os.listdir(data_dir)
    for filename in filenames:
        if not filename.split('.')[1] == 'jpg':
            continue
        img_dir =  os.path.join(data_dir, filename)
        img_dirs.append(img_dir)
    return img_dirs


def json_data(data_dir, json_dir):
    img_dirs = []
    with open(json_dir, "r") as f:
        test_data = json.load(f)
    # check number of items
    img_info_list = test_data['images']
    anno_info_list = test_data['annotations']

    num_len = len(img_info_list)
    for i in tqdm(range(last_save_index, num_len)):
        img_info = img_info_list[i]
        anno_info = anno_info_list[i]
        assert img_info["id"] == anno_info["id"]

        img_name = img_info["file_name"]
        img_dir = os.path.join(data_dir, img_name)
        img_dirs.append(img_dir)

    return img_dirs


def make_json_head():
    json_struct = dict()
    json_struct['images'] = list()
    json_struct['annotations'] = list()
    return json_struct


def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()


last_save_index = 0


def main(data_dir, save_dir):

    APP_ID = '25306081'
    API_KEY = 'hIXaWGPBtGIedfXZxng2PZYU'
    SECRET_KEY = 'IUz9Q8YkUKIB4Mmp4kvvGIVeZboNvC76'

    # client_id 为官网获取的AK， client_secret 为官网获取的SK
    host = 'https://aip.baidubce.com/oauth/2.0/token?' \
           'grant_type=client_credentials&' \
           f'client_id={API_KEY}' \
           f'&client_secret={SECRET_KEY}'
    response = requests.get(host)
    if response:
        print(response.json())

    json_file = make_json_head()

    # img_dir = json_data(data_dir, json_dir)
    img_dirs = image_data(data_dir)
    for i in tqdm(range(last_save_index, len(img_dirs))):
        img_dir = img_dirs[i]
        _, img_name = os.path.split(img_dir)
        img_id = os.path.splitext(img_name)[0]

        img = cv2.imread(img_dir)
        sp = img.shape[0:2]
        if min(sp) < 50:
            img = cv2.resize(img, (int(sp[0]*50/min(sp)), int(sp[1]*50/min(sp))), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(r"G:\test_data\test.jpg", img)
            img_dir = r"G:\test_data\test.jpg"

        """ 读取图片 """
        image = get_file_content(img_dir)
        """ 调用手部关键点识别 """
        client = AipBodyAnalysis(APP_ID, API_KEY, SECRET_KEY)
        hand_json = client.handAnalysis(image)

        # 判断服务是否出错
        if not hand_json.get("error_msg") == None:
            print(hand_json["error_msg"])
            print(f"Note: the json index {i}:{img_name} is not saved")
            break


        image_dict = dict({
            'license': 1,
            'id': img_id,
            'filename': img_name
        })

        if hand_json["hand_num"] == 0:
            value = False
            x_left, y_top, box_w, box_h = 0, 0, 0, 0
            box_score = 0
            coco_kps = np.zeros(63)

        else:
            value = True
            # "[0]"的意义：因为test数据集生成时默认是一个手对应一个json信息，所以默认每个图片仅一个手对象
            # 这里记录的box_h, box_w, x_left, y_top仅为api检测到手掌的位置，并没有乘上缩放因子
            hand_info = hand_json["hand_info"][0]
            location = hand_info["location"]
            box_h = location["height"]
            box_w = location["width"]
            x_left = location["left"]
            y_top = location["top"]
            box_score = location["score"]

            keypoints = list()
            for keyponts_index in range(21):
                hand_parts = hand_info["hand_parts"]
                x_y_score = list()
                keypoint = hand_parts[f"{keyponts_index}"]
                x_y_score.append(keypoint["x"])
                x_y_score.append(keypoint["y"])
                x_y_score.append(keypoint["score"])
                keypoints.append(x_y_score)
            coco_kps = np.array(keypoints)

        coco_kps = coco_kps.flatten().tolist()

        anno_dict = dict({
            'num_keypoints': 21,
            'area': box_h * box_w,
            'iscrowd': 0,
            'keypoints': coco_kps,
            'bbox': [x_left, y_top, box_w, box_h],
            'score': box_score,
            'category_id': 1,
            'hand_value': value,
            'id': img_id

        })

        json_file['images'].append(image_dict)
        json_file['annotations'].append(anno_dict)
        time.sleep(0.4)


        save_path = os.path.join(save_dir, f'baidu_api_index0.json')
        with open(save_path, 'w') as fw:
            json.dump(json_file, fw)
            # print(f"{save_path} have succeed to write")


if __name__ == "__main__":
    data_dir_ = r"G:\test_data\vedio_images\images"
    save_dir_ = r"G:\test_data\vedio_images\anno"
    json_dir_ = r"G:\transmission\anno\v2_2_json\person_keypoints_test2017.json"
    main(data_dir_, save_dir_)