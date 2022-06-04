import json
from tqdm import tqdm
from copy import deepcopy


def update_from_invalid_txt(json_data, ignore_dir, save_dir, debug=False):
    with open(ignore_dir, "r") as f:  # 打开文件
        ignore_info_list = list()
        for ignore_info in f.readlines():
            ignore_info = ignore_info.strip('\n')
            ignore_info_list.append(ignore_info)
        print(f"There are {len(ignore_info_list)} data need to clean")

    imgs_info = json_data['images']
    annos_info = json_data['annotations']

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

        # json数据的删除顺序是从索引号大的开始删
        dele_record.sort(reverse=True)

    # print(dele_record)
    for serial in dele_record:
        del json_data['images'][serial]
        del json_data['annotations'][serial]

    if debug:
        with open(save_dir, "w") as f:
            json.dump(json_data, f)
        print("加载入文件完成...")
        