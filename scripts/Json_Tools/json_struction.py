import json
import time

json_dir_det = r"G:\train_model\pose_hrnet_2_0_bright_diff_w\keypoints_test2017_results_0.json"
json_dir_gt = r"G:\transmission\anno\v2_2_json\person_keypoints_test2017.json"
json_dir_baidu_Api = r"G:\test_data\vedio_images\anno\baidu_api_index0.json"
# G:\test_data\anno\Fix_testdata_mediapipe_cocodt_format.json
# G:\test_data\anno\testdata_mediapipe_cocodt_format.json
mediapipe = r'G:\test_data\anno\Fix_testdata_mediapipe_cocodt_format.json'
baidu = r"G:\test_data\anno\Fix1_testdata_baidu_api_cocodt_format(score0.5).json"



# with open(json_dir_det, "r") as f:
#     det_data = json.load(f)
# print(f"len is {len(det_data)}")
# print(det_data[0].keys())
#
# print("\n")
#


with open(json_dir_gt, "r") as f:
    gt_data = json.load(f)
print(f"This is a dict: {gt_data.keys()}")
img_info = gt_data["images"][0]
anno_info = gt_data["annotations"][0]
print(img_info.keys())
print(anno_info.keys())
# keypoints = anno_info['keypoints']
# print(type(keypoints))
# print(keypoints)



# with open(baidu, "r") as f:
#     baidu_data = json.load(f)
#     print(len(baidu_data))
# with open(mediapipe, "r") as f:
#     mediapipe_data = json.load(f)
#     print(len(mediapipe_data))
# baidu_info = baidu_data[0]
# mediapipe_info = mediapipe_data[0]
# print(f'百度：{baidu_info.keys()}')
# print(f'Media：{mediapipe_info.keys()}')


