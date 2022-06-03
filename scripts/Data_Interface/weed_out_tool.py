'''
    该类可以实现根据crop image的badcase中找出原始（全图）数据中对应的badcase图片名，并生成我们的coco-json格式
    该类在设计时，与多进程并发联动使用
    并且设置了三个使能键：
    1. hard_mode_flag：
        模式使能位——当为1时表示困难模式，当需要筛选的数据是时序数据，往往需要打开困难模式，遍历整个数据集选出，相似度最高的图片
    2. convert_coco_format_flag：
        打包使能位——当为1时表示程序运行时，执行打包工程
    3. keyspoints_check：
        检查使能位——检查crop操作之后，转化后的图片和坐标点是否可以对应上

'''
import os.path
import cv2
import numpy as np
import time
from tools import draw_2d_points
from json_tools import crop_box, _init_save_folder
import json
# HO3D
from ho3d.vis_hand import get_intrinsics, load_pickle_data, projectPoints, canonical_coordinates

class ImageCrop():
    def __init__(self, image_path, save_dir, badcase_resave_dir,
                 image_json_info, annotations_json_info, badcase_num, process_index, mode):
        self.image_path = image_path
        self.save_dir = save_dir
        self.badcase_resave_dir = badcase_resave_dir

        self.image_json_info = image_json_info
        self.annotations_json_info = annotations_json_info
        self.badcase_num = badcase_num
        self.process_index = process_index
        self.mode = mode

        self.have_count = 0
        self.count = 0
        self.fig_num = 21
        self.crop_factor = 2.2
        self.cocobbox_factor = 1.5
        self.min_size = 48
        self.image_dir_list = list()
        self.image_dir = ''
        self.json_file = _init_save_folder()     # 对于没有json文件数据集可以选择直接在数据筛选时生成全图坐标的json文件
        self.hand_keypoints = []

        # 使能标志
        self.hard_mode_flag = 1         # 针对时序数据，设计的困难模式，当标志位为1表示开启困难模式
        self.convert_coco_format_flag = 0       # 打包标志位
        self.keyspoints_check = 0     # 检查crop image与关键点是否对应上
        if self.keyspoints_check:
            cv2.namedWindow('show', cv2.WINDOW_NORMAL)

        self.images = self._get_data()
        assert len(self.images) == len(self.image_dir_list)

    def search(self, badcase_image_dir):
        mark_index = 0
        mark_res = 0
        badcase_image = cv2.imread(badcase_image_dir)
        badcase_img_h, badcase_img_w = badcase_image.shape[:2]
        flag = 0

        record_res_list = list()
        record_index_list = list()

        for index, crop_image_from_whole in enumerate(self.images):
            img_h, img_w = crop_image_from_whole.shape[:2]
            if (badcase_img_h, badcase_img_w) == (img_h, img_w):
                res_img = crop_image_from_whole.astype(np.float32) - badcase_image.astype(np.float32)
                res_img = res_img / (img_w * img_h)

                res = np.sum(np.abs(res_img))
                if res < 10:
                    flag = 1
                    badcase_img_name = os.path.split(badcase_image_dir)[1]
                    if self.hard_mode_flag == 0:
                        mark_index = index
                        mark_res = res
                        break
                    else:
                        record_res_list.append(res)
                        record_index_list.append(index)

        self.have_count += 1
        # if flag == 0:
        #     with open(self.badcase_resave_dir, 'a') as f:
        #         f.write(badcase_image_dir + "\n")
            # print(f'Process {self.process_index}:\t'
            #       f'There is no image to match\t count:{self.count}-{self.have_count}-{self.badcase_num}')

        if flag == 1:
            if self.hard_mode_flag == 0:
                img_dir = self.image_dir_list[mark_index]
                self.images.pop(mark_index)
                self.image_dir_list.pop(mark_index)

            else:
                mark_res = min(record_res_list)
                i = record_res_list.index(mark_res)
                mark_index = record_index_list[i]
                img_dir = self.image_dir_list[mark_index]
                # canve = record_image_list[i]
                self.images.pop(mark_index)
                self.image_dir_list.pop(mark_index)

            self.count += 1

            print(
                f'''
                             - - -- -- ---——————————————————————————————————————--- -- -- - -
                                          {badcase_image_dir}
                             - - -- -- ---——————————————————————————————————————--- -- -- - -''')
            print(f'Process {self.process_index}: {img_dir}:{badcase_img_name} => '
                  f'{mark_res}\t count:{self.count}-{self.have_count}-{self.badcase_num}')

            with open(self.save_dir, 'a') as f:
                f.write(img_dir + '**' + badcase_image_dir + "\n")

    def get_count(self):
        return self.count

    def convert_coco_format(self, hand_keypoints):
        image = cv2.imread(self.image_dir)
        img_h, img_w = image.shape[:2]
        # assert isinstance(img_id, int)

        # 针对HXG数据集进行处理的
        # total_score = np.sum(hand_keypoints[:, -1])
        # if total_score >= 0.6 * 21 and self.mode == 'HXG':
        #     # print(f'Landmark Score is too low! img_path: {img_path}')
        #     print(f'Total score: {total_score}\n')
        #     return 0

        key_pts = hand_keypoints
        kps_valid_bool = key_pts[:, -1].astype(bool)
        key_pts[~kps_valid_bool, :2] = 0

        coco_factor = self.cocobbox_factor
        hand_min = np.min(key_pts, axis=0)  # (2,)
        hand_max = np.max(key_pts, axis=0)  # (2,)
        hand_box_c = (hand_max + hand_min) / 2  # (2, )
        half_size = int(np.max(hand_max - hand_min) * coco_factor / 2.)

        x_left = int(hand_box_c[0] - half_size)
        y_top = int(hand_box_c[1] - half_size)
        x_right = x_left + 2 * half_size
        y_bottom = y_top + 2 * half_size
        box_w = x_right - x_left
        box_h = y_bottom - y_top

        if min(box_h, box_w) < self.min_size:
            return 0

        img_name = os.path.split(self.image_dir)[1]
        file_name = img_name
        img_id = os.path.splitext(img_name)[0]

        current_time = time.localtime()
        record_time = (current_time.tm_year, current_time.tm_mon, current_time.tm_mday,
                       current_time.tm_hour, current_time.tm_min)

        image_dict = dict({
            'license': 1,
            'original_name': img_name,
            'file_name': file_name,
            'coco_url': 'Unavailable',
            'height': img_h,
            'width': img_w,
            'date_captured': record_time,
            'flickr_url': 'Unavailable',
            'image_dir': self.image_dir,
            'id': img_id
        })

        key_pts = key_pts.flatten().tolist()

        # segmentation分别表示左上，右上，右下，左下四个点坐标
        anno_dict = dict({
            'segmentation': [[x_left, y_top, x_right, y_top, x_right, y_bottom, x_left, y_bottom]],
            'num_keypoints': self.fig_num,
            'area': box_h * box_w,
            'iscrowd': 0,
            'keypoints': key_pts,
            'image_id': img_id,
            'bbox': [x_left, y_top, box_w, box_h],  # 1.5 expand
            'category_id': 1,
            'id': img_id
        })

        self.json_file['images'].append(image_dict)
        self.json_file['annotations'].append(anno_dict)

        return 1

    def _get_data(self):
        images_list = list()
        # 遍历whole body=>得到每张图片的crop image
        for index in range(len(self.image_json_info)):
            self._get_image_dir(index)
            if not os.path.exists(self.image_dir):
                print(f'block_index:{self.process_index}')
                print(f'img_dir:{self.image_dir}')
                exit(0)

            img = cv2.imread(self.image_dir)
            _, w = img.shape[:2]

            if self.annotations_json_info is not None:
                if isinstance(self.annotations_json_info[index], dict):
                    assert self.image_json_info[index]['id'] == self.annotations_json_info[index]['image_id']

            flag = self._get_hand_keypoints(index)
            if flag == 0:
                continue

            for hand_index in range(len(self.hand_keypoints)):
                keypoints = self.hand_keypoints[hand_index]

                if self.mode in ['HO3D'] and not np.all(keypoints == 0):
                    keypoints[:, 0] = w - keypoints[:, 0]

                # if self.image_dir == r'G:\test_data\new_data\new_data_from_whole_body_v2_6\images\val2017\000001400051.jpg':
                #     print()
                crop_image_fromwhole, crop_keypoints = crop_box(img, keypoints.copy(), box_factor=self.crop_factor)
                if min(crop_image_fromwhole.shape[0:2]) < self.min_size:
                    continue

                # 检查关键点是否一一对应
                if self.keyspoints_check == 1:
                    print(self.image_dir)
                    im = draw_2d_points(crop_keypoints, crop_image_fromwhole)
                    cv2.imshow('show', im)
                    cv2.waitKey(~self.keyspoints_check+2)
                    # im1 = draw_2d_points(keypoints, img)
                    # cv2.imshow('show', im1)
                    # cv2.waitKey(0)

                if self.mode == 'HO3D':
                    images_list.append(cv2.flip(crop_image_fromwhole, 1))
                else:
                    images_list.append(crop_image_fromwhole)

                self.image_dir_list.append(self.image_dir)

                if self.convert_coco_format_flag == 1:
                    # 写入json
                    self.convert_coco_format(keypoints)

            self.hand_keypoints = []

        print(f"Process {self.process_index}: Finish loaded all crop image!")

        if self.convert_coco_format_flag == 1:
            # 将新的坐标信息写入json存储并释放空间
            json_save_dir = os.path.join(os.path.split(self.save_dir)[0], f'total_{self.process_index}.json')
            with open(json_save_dir, 'w') as fw:
                json.dump(self.json_file, fw)
                print("person_keypoints_test2017.json have succeed to write")
            del self.json_file

        return images_list

    def _get_image_dir(self, index):
        mode = self.mode
        if mode in ['coco', 'new-data']:
            file_name = self.image_json_info[index]['file_name']
            self.image_dir = os.path.join(self.image_path, file_name)
        if mode in ['HFB-train', 'HFB-val']:
            file_name = self.image_json_info[index]['file_name']
            if mode == 'HFB-train':
                middle_name = file_name.split('_')[1]
                self.image_dir = os.path.join(self.image_path, middle_name, file_name)
            if mode == 'HFB-val':
                self.image_dir = os.path.join(self.image_path, file_name)
        if mode in ['HO3D', 'CMU-synth', 'CMU-real', 'CMU-panopticdb', 'RHD', 'FH', 'HXG']:
            self.image_dir = self.image_json_info[index]

    def _get_hand_keypoints(self, index):
        mode = self.mode
        fig_num = self.fig_num
        keypoints_list = list()

        # 这里返回的坐标点格式为：(21,3)
        if mode in ['coco']:
            anno_info = self.annotations_json_info[index]

            hands_kpts = []
            if anno_info["lefthand_valid"]:
                hands_kpts.append(anno_info["lefthand_kpts"])
            if anno_info["righthand_valid"]:
                hands_kpts.append(anno_info["righthand_kpts"])

            for hand_kpts in hands_kpts:
                kp = np.array(hand_kpts).reshape(21, 3)
                kp[:, 2] = 2

                if type(kp) == np.ndarray:
                    if not os.path.exists(self.image_dir) or np.all(kp[2]) == 0:
                        return 0

                self.hand_keypoints.append(kp)

            return 1


            # for hand_index in range(2):
            #     hand_keypoints = np.zeros([fig_num, 3])
            #     if hand_index == 0:
            #         hand_keypoints = np.array(self.annotations_json_info[index]['lefthand_kpts']).reshape(21, 3)
            #     else:
            #         hand_keypoints = np.array(self.annotations_json_info[index]['righthand_kpts']).reshape(21, 3)
            #     keypoints_list.append(hand_keypoints)

        if mode in ['HFB-train', 'HFB-val']:
            keypoints = np.array(self.annotations_json_info[index]['keypoints']).reshape(136, 3)

            for hand_index in range(2):
                hand_keypoints = np.zeros([fig_num, 3])
                if hand_index == 0:
                    hand_keypoints = keypoints[94:115, :]
                else:
                    hand_keypoints = keypoints[115:136, :]

                if np.sum(hand_keypoints[:, 2]) > 21:
                    self.hand_keypoints.append(hand_keypoints)
            return 1

        if mode in ['HO3D']:
            image_dir = self.image_dir
            img_name = os.path.basename(image_dir)
            path = image_dir[:image_dir.find('rgb')-1]      # ./train/AP10
            data_name = os.path.basename(path)  # AP10
            id = path[-1]
            inter_param_path = os.path.join(self.image_path, "calibration", data_name[:-1],
                                            "calibration", 'cam_' + id + '_intrinsics.txt')

            pickle_path = os.path.join(path, "meta", img_name.split(".")[0] + ".pkl")
            # get Camera internal parameters
            K = get_intrinsics(inter_param_path).tolist()
            # print(img_dir)
            opt_pick_data = load_pickle_data(pickle_path)
            hand_joints_3d = opt_pick_data['handJoints3D']

            if type(hand_joints_3d) is np.ndarray and np.all(hand_joints_3d != 0) and hand_joints_3d.shape == (21, 3):
                hand_keypoints = projectPoints(hand_joints_3d, K)
                hand_keypoints = canonical_coordinates(hand_keypoints)
            else:
                hand_keypoints = np.zeros([fig_num, 3])

            keypoints_list.append(hand_keypoints)

        if mode in ['CMU-synth', 'CMU-real']:
            image_dir = self.image_dir
            path, image_name = os.path.split(image_dir)
            json_dir = os.path.join(path, os.path.splitext(image_name)[0]+'.json')
            with open(json_dir, 'r')as f:
                json_info = json.load(f)
                if type(json_info['hand_pts']) == list:
                    hand_keypoints = np.array(json_info['hand_pts'])
                    self.hand_keypoints.append(hand_keypoints)
            return 1

        if mode in ['CMU-panopticdb', 'RHD', 'FH', 'HXG', 'new-data']:
            if isinstance(self.annotations_json_info[index], list):     # hxg
                self.hand_keypoints = self.annotations_json_info[index]
            else:
                hand_keypoints = self.annotations_json_info[index]
                self.hand_keypoints.append(hand_keypoints)

            return 1
            # return keypoints_list



