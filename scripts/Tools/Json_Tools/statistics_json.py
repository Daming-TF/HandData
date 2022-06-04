import json
import copy

def main(json_path):
    YT3D_num = HF_num = HXG_num = FH_num = CMUreal_num = \
        CMUpanopticDB_num = CMUsynth_num = RHD_num = coco_num = HO3D_num = Ours1_num = Ours2_num = 0
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
        print(f"The json path is:{json_path}")
        print(len(imgs_info))
        print(len(annos_info))

        dele_record = list()
        num_imgs = len(imgs_info)
        # 生成0-num_imgs数字列表
        for i in range(num_imgs):
            img_info = imgs_info[i]
            anno_info = annos_info[i]
            # img_info['id'] ！= anno_info['id']情况触发异常
            # assert img_info['id'] == anno_info['image_id']
            id = int(img_info['id'])
            if id >= 300_000 and  id < 400_000:
                YT3D_num = YT3D_num + 1
            elif id >= 400_000 and  id < 500_000:
                HF_num += 1
            elif id >= 500_000 and  id < 600_000:
                HXG_num += 1
            elif id >= 600_000 and  id < 800_000:
                FH_num += 1
            elif id >= 800_000 and  id < 900_000:
                CMUreal_num += 1
            elif id >= 900_000 and  id < 1_000_000:
                CMUpanopticDB_num += 1
            elif id >= 1_000_000 and  id < 1_100_000:
                CMUsynth_num += 1
            elif id >= 1_100_000 and  id < 1_200_000:
                RHD_num += 1
            elif id >= 1_200_000 and  id < 1_300_000:
                coco_num += 1
            elif id >= 1_300_000 and  id < 1_400_000:
                HO3D_num += 1
            elif id >= 1_400_000 and  id < 1_500_000:
                Ours1_num += 1
            elif id >= 1_500_000 and id < 1_600_000:
                Ours2_num += 1
        print(f"YT3D_num is: {YT3D_num}")
        print(f"HF_num is: {HF_num}")
        print(f"HXG_num is: {HXG_num}")
        print(f"FH_num is: {FH_num}")
        print(f"CMUreal_num is: {CMUreal_num}")
        print(f"CMUpanopticDB_num is: {CMUpanopticDB_num}")
        print(f"CMUsynth_num is: {CMUsynth_num}")
        print(f"RHD_num is: {RHD_num}")
        print(f"coco_num is: {coco_num}")
        print(f"HO3D_num is: {HO3D_num}")
        print(f"ours1_num is: {Ours1_num}")
        print(f"ours2_num is: {Ours2_num}")

# hand_labels  hand_labels_synth   hand143_panopticdb
if __name__ == "__main__":
    json_path_ = \
        r"E:\v2_6\annotations\person_keypoints_val2017.json"

    # G:\test_data\new_data\person_keypoints_test2017(update).json
    # G:\test_data\new_data\new_data_from_whole_body\annotations\person_keypoints_test2017-update.json

    main(json_path_)