import json

from library.json_tools import load_json_data, make_json_head, get_ids


def main():
    json_path = r'E:\left_hand_label_data\annotations\v2_6\test_data_add_hand\mediapipe-detecter\badcase_test.json'
    save_path = r'E:\left_hand_label_data\annotations\v2_6\test_data_add_hand\mediapipe-detecter\badcase_test-convert.json'
    images_dict, annotations_dict = load_json_data(json_path)

    write_json(images_dict, annotations_dict, save_path)


def write_json(images_dict, annotations_dict, save_path):
    json_head = make_json_head()
    images_ids = get_ids(images_dict)
    for image_id in images_ids:
        image_info = images_dict[image_id]

        annotation_info_list = annotations_dict[image_id]
        for annotation_info in annotation_info_list:
            json_head["images"].append(image_info)
            json_head["annotations"].append(annotation_info)

    with open(save_path, 'w') as f:
        json.dump(json_head, f)


if __name__ == '__main__':
    main()
