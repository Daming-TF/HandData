import json
import os
import copy
from tqdm import tqdm

def main(ignore_path, json_path, save_path):
    with open(ignore_path, "r") as f:  # 打开文件
        ignore_info_list = list()
        for ignore_info in f.readlines():
            ignore_info = ignore_info.strip('\n')
            ignore_info_list.append(ignore_info)

    # json文件结构(字典结构)包含：'info'-dict, 'licenses'-list(dict),
    # 'categories'-list(dict), 'images'-list(dict), 'annotations'-list(dict)
    with open(json_path, "r") as f:
        json_data = json.load(f)
        jsonfile = copy.deepcopy(json_data)
        # 'images'结构(列表结构)包含多个字典对象，每个对象结构如下：
        # 'license', 'file', 'coco_url', 'height',
        # 'width', 'date_captured', 'flickr_url', 'id'
        imgs_info = jsonfile['images']
        # 'annotations'结构(列表结构)包含多个字典对象，每个对象结构如下：
        # 'segmentation', 'num_keypoints', 'area', 'iscrowd',
        # 'keypoints', 'image_id', 'bbox', 'category', 'id'
        annos_info = jsonfile['annotations']

        dele_record = list()
        num_imgs = len(imgs_info)
        # 生成0-num_imgs数字列表
        for i in tqdm(range(num_imgs)):
            img_info = imgs_info[i]
            anno_info = annos_info[i]
            # img_info['id'] ！= anno_info['id']情况触发异常
            assert img_info['id'] == anno_info['id']
            for ignore_info in ignore_info_list:
                ignore_info = ignore_info.split('**')[0]
                img_name = os.path.basename(ignore_info)
                id = str(int(img_name.split('.')[0]))
                a = f"{json_data['images'][i]['id']}"
                b = json_data['annotations'][i]['id']
                if id == f"{json_data['images'][i]['id']}":
                    # del json_data['images'][i]
                    if id == f"{json_data['annotations'][i]['id']}":
                    # del json_data['annotations'][i]
                        # print(id)
                        dele_record.append(i)

        dele_record = list(reversed(dele_record))
        # print(dele_record)
        for serial in dele_record:
            del json_data['images'][serial]
            del json_data['annotations'][serial]

    with open(save_path, "w") as f:
        json.dump(json_data, f)
    print("加载入文件完成...")


if __name__ == "__main__":
    ignore_path_ = \
        r"G:\test_data\new_data\new_data_from_whole_body\weed_out_badcase\whold-body-badcase.txt"
    json_path_ = \
        r"G:\test_data\new_data\new_data_from_whole_body\annotations\total.json"
    save_path_ = r"G:\test_data\new_data\new_data_from_whole_body\annotations\test.json"
    main(ignore_path_, json_path_, save_path_)