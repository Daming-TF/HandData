import json
from tqdm import tqdm
import time

def main():
    # G:\test_data\debug\person_keypoints_test2017.json
    # testdata_baidu_api.json
    # testdata_mediapipe.json
    json_dir = r"G:\test_data\vedio_images\anno\testdata_mediapipe_img.json"
    save_path = r"G:\test_data\vedio_images\anno\testdata_mediapipe_img_cocodt_format.json"
    # “image_id”, "category_id", “keypoints”, “score”
    with open (json_dir, "r") as f:
        json_data = json.load(f)

    annoes_info = json_data["annotations"]

    data_pack = []


    for index in tqdm(range(len(annoes_info))):
        data = {}
        anno = annoes_info[index]
        image_id = anno["id"]
        category_id = anno["category_id"]
        keypoints = anno["keypoints"]
        # print(keypoints)
        print(type(keypoints))

        # key_points = np.zeros(
        #         (_key_points.shape[0], self.num_joints * 3), dtype=np.float
        #     )

        data['image_id'] = image_id
        data['category_id'] = category_id
        data['keypoints'] = keypoints
        data['score'] = 0.5
        data_pack.append(data)

    with open(save_path, 'w') as fw:
        json.dump(data_pack, fw)
    print(f'SUCCESS')

if __name__ == "__main__":
    main()