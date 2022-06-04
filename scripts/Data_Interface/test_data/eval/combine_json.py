import json
import os

from library.json_tools import make_json_head


def combine_json(json_file, json_dir):
    with open(json_dir, "r") as f:
        data = json.load(f)
    # check number of items
    num_imgs = len(data['images'])
    num_annos = len(data['annotations'])
    assert num_imgs == num_annos
    print(f'{os.path.basename(json_dir)} load SUCCESS!')

    for i in range(num_imgs):
        json_file['images'].append(data['images'][i])
        json_file['annotations'].append(data['annotations'][i])
    print(f'Finish to write json --{os.path.basename(json_dir)}')


def main():
    suffix = 'v3-align'
    json_path = r'E:\test_data\test_data_from_whole_body\annotations\coco_eval\dt'
    vedio_name = ["hand_test_01", "hand_test_02", "hand_test_03", "hand_test_04", "hand_test_05", "hand_test_06",
                  "hand_test_07", "hand_test_08", "hand_test_09", "hand_test_10"]
    save_dir = fr'E:\test_data\test_data_from_whole_body\annotations\{suffix}.json'


    json_head = make_json_head()

    for i in range(len(vedio_name)):
        json_name = vedio_name[i]
        json_dir = os.path.join(json_path, json_name, json_name+f'_{suffix}.json')
        combine_json(json_head, json_dir)

    with open(save_dir, 'w')as f:
        json.dump(json_head, f)
        print("success!")


if __name__ == '__main__':
    main()
