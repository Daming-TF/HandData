import json
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

        for index in tqdm(range(len(ignore_info_list))):
            ignore_info = ignore_info_list[index]
            for i in range(num_imgs):
                img_info = imgs_info[i]
                anno_info = annos_info[i]
                assert img_info['id'] == anno_info['id']
                # 只图片上有一只手是错的，都把两只手看作是错的
                img_dir = img_info['image_dir']
                if ignore_info == img_dir:
                    dele_record.append(i)

        print(dele_record)
        print(len(dele_record))
        dele_record.sort(reverse=True)
        print(dele_record)

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
        r'G:\test_data\new_data\new_data_from_whole_body\total.json'
    save_path_ = r"G:\test_data\new_data\new_data_from_whole_body\total-update.json"
    main(ignore_path_, json_path_, save_path_)