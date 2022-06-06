import json
import copy

import numpy as np

json_dir = r"G:\test_data\debug\testdata_baidu_api_cocodt_format.json"
save_dir = r"G:\test_data\debug\FIX(1pic)-testdata_baidu_api_cocodt_format.json"

def main():
    with open(json_dir, "r") as f:
        json_data = json.load(f)
    data_pack = copy.deepcopy(json_data)
    pack = []
    for data in data_pack:
        keypoints = np.array(data['keypoints'])
        if np.all(keypoints == 0):
            data['keypoints'] = list(np.zeros(63))
        pack.append(data)
    with open(save_dir, "w") as f:
        json.dump(pack, f)
    print("scessed!")



if __name__ == "__main__":
    main()