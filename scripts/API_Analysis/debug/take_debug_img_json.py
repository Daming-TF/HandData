import json
# person_keypoints_test2017
# testdata_mediapipe
# testdata_baidu_api
json_dir = r"G:\test_data\anno\testdata_baidu_api.json"
save_path = r"G:\test_data\debug\testdata_baidu_api.json"
meg = [ 729287]  # 728859,  728886, 729287,729301, 729750

def main():
    data_pack = {}
    img_list, anno_list = [], []
    with open (json_dir, 'r') as f:
        data = json.load(f)
    img_infos = data["images"]
    anno_infos = data["annotations"]
    for index in range(len(img_infos)):
        img_info = img_infos[index]
        anno_info = anno_infos[index]
        id = int(img_info['id'])
        if id in meg:
            assert img_info['id'] == anno_info['id']
            img_list.append(img_info)
            anno_list.append(anno_info)
    data_pack["images"] = img_list
    data_pack["annotations"] = anno_list
    data_pack["categories"] = data["categories"]

    with open(save_path, 'w') as fw:
        json.dump(data_pack, fw)
    print("SCESSED!")


if __name__ == "__main__":
    main()