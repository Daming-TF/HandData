import json
import os
import requests
import numpy as np
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--json_folder",
                    default=fr"F:\image\CMU\hand_labels_synth\hand_labels_synth\badcase\upload json",
                    help="json folder that has corresponding json files")

opts = parser.parse_args()



def get_label_feature(json_path):
    with open(json_path, 'r') as f:
        json_data = json.load(f)
        anno_info = json_data['annotations'][0]
        keypoints = anno_info['keypoints']

        hand_points = dict()
        for i, keypoint in enumerate(keypoints):
            preds = np.array(keypoint).reshape(21, 2)
            new_preds = np.ones((21, 3)) * 2
            new_preds[:, :2] = preds.copy()

            for index in range(new_preds.shape[0]):
                hand_points[f'{i}-{index}'] = list(new_preds[index])
    return [hand_points]


class Requester(object):
    def __init__(self):
        self.taskID = 6131
        self.header = {"X-APPID": "ZHIXU", "X-APPKEY": "94KhTW0yOcCKwJ4F"}
        self.upload_path = "http://ai.huya.com/annotation/api/v3/file/uploadFileFromApi"
        self.update_path = "http://ai.huya.com/annotation/api/v3/file/updateSample"
        self.retry_time = 10
        self.timeout = 25

    def upload(self, file, is_img):
        retry_time = self.retry_time
        if is_img:
            host = self.upload_path
        else:
            host = self.update_path

        while retry_time > 0:
            try:
                if is_img:
                    response = requests.post(host,
                                             data={'taskID': self.taskID},
                                             files=file,
                                             headers=self.header,
                                             timeout=self.timeout,
                                             ).json()
                else:
                    response = requests.put(host,
                                            data=file.encode("utf-8"),
                                            headers=self.header).json()

                if response['code'] == 200:
                    return response
                else:
                    print(f"response['code']: {response['code']}")
                    return -1
            except requests.exceptions.Timeout:
                print(f' [!] Timeout, repeat {self.retry_time - retry_time + 1}th ...')
            except Exception as ex:
                print(f' [!] {ex}, repeat {self.retry_time - retry_time + 1}th ...')

            retry_time -= 1

        print(" [!] Encounter max retry and skip it!")
        return -1


def main():
    # 数据上传初始化设置
    requester = Requester()

    # 数据路径和注释路径分别排序存储
    # img_paths = sorted([os.path.join(opts.img_folder, file) for file in os.listdir(opts.img_folder)
    #                     if file.endswith('.jpg') or file.endswith('.png') or file.endswith('.jpeg')])
    # json_paths = sorted([os.path.join(opts.json_folder, file) for file in os.listdir(opts.json_folder)
    #                      if file.endswith('.json')])

    json_paths = [os.path.join(opts.json_folder, file) for file in os.listdir(opts.json_folder)
                  if file.endswith('.json')]
    img_paths = []
    for json_dir in json_paths:
        with open(json_dir, 'r')as f:
            json_data = json.load(f)
            img_dir = json_data['images'][0]['file_name']
            img_paths.append(img_dir)

    assert len(img_paths) == len(json_paths), print("Number of images and number of json-files are different!")

    for img_path, json_path in zip(img_paths, json_paths):
        # Uploading img file first
        with open(json_path, 'r') as f:
            json_data = json.load(f)
            mode = json_data['images'][0]['mode']
        upload_name = 'CMUreal_' + mode + "_" + os.path.basename(img_path)
        files = {'file': (upload_name, open(img_path, 'rb'))}
        response = requester.upload(files, is_img=True)

        if response != -1 and response['code'] == 200:
            print(f" [*] Uploade {os.path.basename(img_path)} IMG success!")

            # if img was uploaded successly, next upload json file
            label_feature = get_label_feature(json_path)

            files = {'sampleID': int(response['msg']),
                     'labelFeature': label_feature,
                     'originalSampleID': ''}
            files = json.dumps(files, ensure_ascii=False)
            response = requester.upload(files, is_img=False)

            if response != -1 and response['code'] == 200:
                print(f" [*] Upload {os.path.basename(json_path)} JSON success!")
            else:
                print(f' [!] Upload {os.path.basename(json_path)} failed, go to next sample...')

        else:
            print(f' [!] Upload {os.path.join(img_path)} failed, go to next sample...')


if __name__ == "__main__":
    main()
