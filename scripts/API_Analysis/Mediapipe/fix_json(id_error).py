import json

json_dir = r"G:\test_data\vedio_images\anno\testdata_mediapipe_img_cocodt_format.json"
save_dir = r"G:\test_data\vedio_images\anno\FIX-testdata_mediapipe_img_cocodt_format.json"

data_pack = []

with open(json_dir, "r") as f:
    annoes_info = json.load(f)

for index in range(len(annoes_info)):
    data = {}
    anno = annoes_info[index]
    image_id = int(anno["image_id"])
    category_id = anno["category_id"]
    keypoints = anno["keypoints"]
    # print(keypoints)
    print(image_id)

    # key_points = np.zeros(
    #         (_key_points.shape[0], self.num_joints * 3), dtype=np.float
    #     )

    data['image_id'] = image_id
    data['category_id'] = category_id
    data['keypoints'] = keypoints
    data['score'] = 0.5
    data_pack.append(data)

with open(save_dir, 'w') as fw:
    json.dump(data_pack, fw)
print(f'SUCCESS')