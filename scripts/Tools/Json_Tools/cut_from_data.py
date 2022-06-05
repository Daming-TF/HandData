"""
功能：读取json_path对应的json文件，
    把指定image id∈(end_index, data_index)范围的json数据抽离出来，
    并以新的数据结构体保存在save_path
"""
from library.json_tools import make_json_head
import json
import os
from tqdm import tqdm

data_index = 1_000_000
end_index = 1_100_000


def main(json_path, save_path):
    file_path = os.path.split(save_path)[0]
    is_exists = os.path.exists(file_path)
    if not is_exists:
        os.mkdir(file_path)
    json_file = make_json_head()

    with open(json_path, "r") as f:
        data = json.load(f)
        print(f'{os.path.basename(json_path)} load SUCCESS!')

    images_data = data['images']
    annos_data = data['annotations']

    for index in tqdm(range(len(images_data))):
        image_id = images_data[index]['id']
        anno_id = annos_data[index]['id']
        assert image_id == anno_id

        if image_id < data_index or image_id >= end_index:
            json_file['images'].append(data['images'][index])
            json_file['annotations'].append(data['annotations'][index])

    print(f"dele num is {len(images_data)-len(json_file['images'])}")
    # print(f"This data has >>{len(json_file['images'])}<<  pic")

    with open(save_path, 'w') as fw:
        json.dump(json_file, fw)
    print(f'{os.path.split(save_path)[1]} write succes!')


if __name__ == "__main__":
    json_path = r'G:\transmission\anno\Test_screening\保留积极影响数据集\person_keypoints_train2017.json'
    save_path = r'G:\transmission\anno\Test_screening\保留积极影响数据集\person_keypoints_train2017.json'
    main(json_path, save_path)
