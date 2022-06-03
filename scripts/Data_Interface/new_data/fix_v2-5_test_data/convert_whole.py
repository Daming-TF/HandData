"""
从原本的v2_5test数据，重新覆盖“images”：‘id’    “annotations”：‘id’,'image_id'
并重新保存图片
"""
import json
from tqdm import tqdm

from json_tools import make_json_head
from convert_tools import convert_coco_format_for_whole

def main():
    json_head = make_json_head()
    json_path = r'G:\test_data\new_data\new_data_from_whole_body\annotations\person_keypoints_test2017-update.json'
    save_dir = r'G:\test_data\new_data\new_data_from_whole_body\new_images\test2017'
    json_save_path = r'G:\test_data\new_data\new_data_from_whole_body\annotations\test-update.json'
    start_image_id = 1400000

    with open(json_path, 'r')as f:
        json_data = json.load(f)
        images = json_data['images']
        annotations = json_data['annotations']

    image_id = start_image_id
    coco_id = 1
    while 1:
        two_hand_flag = 0
        iter_num = len(images)
        print(f"Now we have finish:{image_id-start_image_id} \tRemain:{iter_num}")

        init_image_info = images[0]
        init_annotation_info = annotations[0]
        if image_id == 1401504:
            print()
        convert_coco_format_for_whole(json_head, init_image_info, init_annotation_info, save_dir,
                                      coco_id=coco_id, image_id=image_id)

        init_image_dir = init_image_info['image_dir']
        for i in range(1, iter_num):
            search_image_info = images[i]
            search_annotation_info = annotations[i]

            search_image_dir = search_image_info['image_dir']
            if init_image_dir == search_image_dir:
                convert_coco_format_for_whole(json_head, search_image_info, search_annotation_info, save_dir,
                                              save_flag=0, coco_id=coco_id, image_id=image_id)
                two_hand_flag = 1
                break

        image_id += 1
        coco_id += 1

        if two_hand_flag:
            del images[i]
            del annotations[i]
        del images[0]
        del annotations[0]

        if len(images)==0:
            break

    with open(json_save_path, 'w')as f:
        json.dump(json_head, f)


if __name__ == '__main__':
    main()