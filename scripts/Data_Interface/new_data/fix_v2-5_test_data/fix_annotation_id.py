import json
from tqdm import tqdm
from json_tools import make_json_head

json_path = r'G:\test_data\new_data\new_data_from_whole_body\annotations\person_keypoints_test2017-update.json'
save_path = r'G:\test_data\new_data\new_data_from_whole_body\annotations\person_keypoints_test2017-update_id.json'


def main():
    json_head = make_json_head()
    with open(json_path, 'r')as f:
        json_data = json.load(f)
        images = json_data['images']
        annotations = json_data['annotations']

    iter_num = len(annotations)

    for i in tqdm(range(iter_num)):
        image_info = images[i]
        annotation_info = annotations[i]

        assert image_info['id']==annotation_info['image_id']
        annotation_info['id']=i+1

        json_head['images'].append(image_info)
        json_head['annotations'].append(annotation_info)

    with open(save_path, 'w')as sf:
        json.dump(json_head, sf)


if __name__ == '__main__':
    main()