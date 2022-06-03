import json
from json_tools import make_json_head
from tqdm import tqdm
import os
import copy


def main():
    json_dir = r'E:\test_data\test_data_from_whole_body\annotations\need_to_remark.json'
    save_path = r'E:\test_data\test_data_from_whole_body\upload_remark'
    json_head = make_json_head()
    with open(json_dir, 'r')as f:
        json_data = json.load(f)
        images = json_data['images']
        annotations = json_data['annotations']

    iter_num = len(images)

    for i in tqdm(range(iter_num)):
        json_model = copy.deepcopy(json_head)
        image_info = images[i]
        annotation_info = annotations[i]
        image_dir = image_info['image_dir']
        image_id = os.path.basename(image_dir).split('.')[0]
        save_dir = os.path.join(save_path, image_id+'.json')

        json_model['images'].append(image_info)
        json_model['annotations'].append(annotation_info)

        with open(save_dir, 'w')as f:
            json.dump(json_model, f)

        del json_model


if __name__ == '__main__':
    main()