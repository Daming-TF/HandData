"""
采集test数据集之后，有些场景detector效果差，一些场景肢体关键点差，所以通过该程序可以融合他们优点
"""
import cv2
import copy
import json
import numpy as np
from tqdm import tqdm

from library.json_tools import _init_save_folder, crop_box
from library.tools import draw_2d_points


# v3pose_json_dir = r'E:\test_data\test_data_from_whole_body\annotations\v3-pose-video.json'
# v3det_json_dir = r'E:\test_data\test_data_from_whole_body\annotations\v3-detector-video.json'
# save_dir = r'E:\test_data\test_data_from_whole_body\annotations\v3.json'
v3pose_json_dir = r'E:\test_data\test_data_from_whole_body\annotations\v3-video.json'
v3det_json_dir = r'E:\test_data\test_data_from_whole_body\annotations\mediapipe-video.json'
save_dir = r'E:\test_data\test_data_from_whole_body\annotations\average_pseudo_labels.json'
debug = 1


def check_same_hand(hands_list):
    l_prieds = np.array(hands_list[0]).reshape(21, 3)
    r_prieds = np.array(hands_list[1]).reshape(21, 3)
    sum = np.sum(np.abs(l_prieds - r_prieds))
    if sum < 500:
        hands_list[1] = [0]*21*3
    return hands_list


def handslist_to_image(image, hands_list):
    img = copy.deepcopy(image)
    for index in range(len(hands_list)):
        prieds = np.array(hands_list[index]).reshape(21, 3)
        img = draw_2d_points(prieds, img)
    return img


def exchange(index):
    if index == 0:
        res = 1
    elif index == 1:
        res = 0
    else:
        print("index just only could chose in [0,1]")
        exit(1)
    return res


def compare_object(hand_prieds_pose, det_keypoints, th=8):
    '''

    Parameters
    ----------
    hand_prieds_pose:   手21个关键点坐标 {numpy 21*3}
    det_keypoints：  通过det检测的关键点列表 {list 2}-每个单元是63*1(21个坐标点)

    Returns
    -------
    res: 匹配的索引号
    diff：对应匹配索引号的关键点坐标差值

    '''
    diff = 1000000
    res = None
    hand_prieds_pose[:, 2] = 2
    size = crop_box(hand_prieds_pose)
    # 转化坐标
    for index, det_prieds in enumerate(det_keypoints):
        det_prieds = np.array(det_prieds).reshape(21, 3)
        sum = np.sum(np.abs(det_prieds - hand_prieds_pose))/size

        if sum < diff and sum < th:     # 匹配的时候一般小于1
            diff = sum
            res = index
    print(f'Diff: {diff}')
    return res, diff


def main():
    cv2.namedWindow('show', cv2.WINDOW_NORMAL)
    json_hand = _init_save_folder()
    with open(v3pose_json_dir, 'r')as f:
        pose_data = json.load(f)
        pose_images = pose_data['images']
        pose_annotations = pose_data['annotations']

    with open(v3det_json_dir, 'r')as f:
        det_data = json.load(f)
        det_images = det_data['images']
        det_annotations = det_data['annotations']

    assert (len(pose_images) == len(pose_annotations) == len(det_images) == len(det_annotations))
    iter_nun = len(pose_images)
    for i in range(iter_nun):
        # if i < 15000:        # 1168
        #     continue
        pose_images_info = pose_images[i]
        pose_annotations_info = pose_annotations[i]
        pose_keypoints = pose_annotations[i]['keypoints']
        det_keypoints = det_annotations[i]['keypoints']

        mix_image_info = copy.deepcopy(pose_images_info)
        mix_keypoints_info = copy.deepcopy(pose_annotations_info)#

        img_dir = pose_images_info['image_dir']
        image_id = pose_images_info['id']
        image = cv2.imread(img_dir)

        print(f'count:{i}\tid:{image_id}')

        # 这里很关键，mix_landmarks是一个初始化为2*1*1的列表，mix_landmarks[0]记录左手，[1]记录右手
        # 这和跑肢体检测器输出坐标格式是一致的
        mix_landmarks = [[], []]
        standard_hands, match_hands = [], []
        if not (np.all(np.array(pose_keypoints) == 0) or np.all(np.array(det_keypoints) == 0)):
            if len(pose_keypoints) >= len(det_keypoints):
                standard_hands, match_hands = pose_keypoints, det_keypoints
            else:
                standard_hands, match_hands = det_keypoints, pose_keypoints

            for i, keypoints in enumerate(standard_hands):
                prieds = np.array(keypoints).reshape(21, 3)
                if np.all(prieds == 0):
                    mix_landmarks[i] = match_hands[i]
                    continue
                index, _ = compare_object(prieds, match_hands)
                if index == None:       # 表示match_hands中没有匹配的手
                    mix_landmarks[i] = standard_hands[i]
                    continue

                # 如果keypoints是非零坐标&有匹配的手
                standard_prieds = np.array(standard_hands[i]).reshape(21, 3)
                match_prieds = np.array(match_hands[index]).reshape(21, 3)
                mix_prieds = (standard_prieds + match_prieds)/2
                mix_landmarks[i] = mix_prieds.flatten().tolist()

        # 肢体检测器或者手检测器至少有一个检测器没有检测到手的，全为空的
        else:
            if np.all(np.array(pose_keypoints) == 0):
                mix_landmarks = det_keypoints
            if np.all(np.array(det_keypoints) == 0):
                mix_landmarks = pose_keypoints

        # 这里主要解决的是det的标签混乱导致，同一个手，pose和det的标签不同
        mix_landmarks = check_same_hand(mix_landmarks)

        if debug:
            img_pose = handslist_to_image(image, pose_keypoints)
            img_det = handslist_to_image(image, det_keypoints)
            img_mix = handslist_to_image(image, mix_landmarks)
            canve1 = np.hstack([img_pose, img_det])
            canve2 = np.hstack([canve1, img_mix])
            cv2.imshow('show', canve2)
            cv2.waitKey(1)

        mix_keypoints_info['keypoints'] = mix_landmarks
        json_hand['images'].append(mix_image_info)
        json_hand['annotations'].append(mix_keypoints_info)

    with open(save_dir, 'w')as f:
        json.dump(json_hand, f)
    print(f'Successed to write in {save_dir}')


if __name__ == '__main__':
    main()
