"""
重新构建v2_6的数据集，一张图片对应一个images的一个unit，每个手对应annotations的一个unit
images['id']标识图片id
annotations['id']标识手对象id
annotation['image_id']标识图片id

同时把所有关键点坐标保留两位保存

对于不同数据集的convert_coco_format仅需要对load_data做修改
对于load_data: 参照coco的读取数据方式，以原始图片名做索引号，对每张图片存储关键信息单元如下
unit_dict = dict({
                'hand_type': hand_type,
                'image_path': img_path,
                'keypoints': keypoints,
            })

YouTu 3D数据集原始数据全部都给了标注团队标注，所以对于该数据集主要输入数据是标注团队反馈回来的json文件

"""
import argparse
import os
from tqdm import tqdm
import cv2
import numpy as np
import json
import copy
from collections import defaultdict
import time

from json_tools import _init_save_folder
from dataset_v2_6 import convert_coco_format_from_wholebody
from tools import draw_2d_points

COCOBBOX_FACTOR = 1.5
COCO_START_ID = 600_000
VAL_NUM = 0
Divide_1 = COCO_START_ID + VAL_NUM
Debug = 0


def main(data_dir, save_dir):
    # init
    json_file_model = _init_save_folder(save_dir)
    val_json, train_json = copy.deepcopy(json_file_model), copy.deepcopy(json_file_model)
    coco_id, image_id = COCO_START_ID, COCO_START_ID
    image_counter = 0

    # Load information using coco method
    print("loading the data......")
    data_dict, tag_names = load_data(data_dir)

    img_num = len(tag_names)
    shuffle_list = np.arange(img_num)
    rng = np.random.default_rng(12345)
    rng.shuffle(shuffle_list)

    print("processing data......")
    for index in tqdm(range(img_num)):
        shuffle_id = shuffle_list[index]
        tag_name = tag_names[shuffle_id]

        unit_list = data_dict[tag_name]
        if Debug:
            debug_image = cv2.imread(unit_list[0]['image_path'])

        flag = 0
        for hand_index, unit_dict in enumerate(unit_list):
            img_path = unit_dict['image_path']
            keypoints = unit_dict['keypoints']

            if VAL_NUM > image_counter >= 0:
                mode = 'val2017'
                json_file = val_json
            elif image_counter >= VAL_NUM:
                if image_counter == VAL_NUM and VAL_NUM != 0:
                    with open(os.path.join(save_dir, 'annotations', f'person_keypoints_val2017.json'), 'w') as fw:
                        json.dump(val_json, fw)
                mode = 'train2017'
                json_file = train_json

            flag = convert_coco_format_from_wholebody(img_path, keypoints, json_file, mode, save_dir, coco_id,
                                                      image_id, save_flag=flag)
            if Debug:
                debug_image = draw_2d_points(keypoints, debug_image)

            if flag == 1:
                coco_id +=1

        if Debug:
            cv2.imshow('show', debug_image)
            cv2.waitKey(0)

        image_id += 1
        if flag == 1:
            image_counter += 1

    print(f"Writing the json file person_keypoints_train.json......")
    with open(os.path.join(save_dir, 'annotations', f'person_keypoints_train2017.json'), 'w') as fw:
        json.dump(train_json, fw)


def load_data(data_dir, versions=['gs', 'hom', 'sample', 'auto']):
    data_dict = defaultdict(list)

    num_freqs = db_size('training')
    num_total_imgs = 4 * num_freqs  # 所有数据数量
    indexs = np.arange(num_total_imgs)  # index是对应每张图片的序号的列表
    np.random.shuffle(indexs)

    # load annotations
    # BASE_PATH = r"D:\Data\landmarks\FreiHAND_pub_v2\training"
    db_data_anno = list(load_db_annotation(data_dir, 'training'))

    for i in tqdm(range(indexs.shape[0])):
        keypoints = np.ones([21, 3])
        index = indexs[i]
        version = versions[index // num_freqs]

        # load image and mask
        image_path = read_img_cv(index, data_dir, 'training', version)

        # annotation for this frame
        K, mano, xyz = db_data_anno[index % num_freqs]
        K, mano, xyz = [np.array(x) for x in [K, mano, xyz]]
        keypoints[:, :2] = projectPoints(xyz, K)


        unit_dict = dict({
            'image_path': image_path,
            'keypoints': keypoints,
        })

        tag_name =os.path.basename(image_path)
        data_dict[tag_name].append(unit_dict)

    return data_dict, list(data_dict.keys())


""" Dataset related functions. """
def db_size(set_name):
    """ Hardcoded size of the datasets. """
    if set_name == 'training':
        return 32560  # number of unique samples (they exists in multiple 'versions')
    elif set_name == 'evaluation':
        return 3960
    else:
        assert 0, 'Invalid choice.'


def load_db_annotation(base_path, set_name=None):       # BASE_PATH, 'training'
    if set_name is None:
        # only training set annotations are released so this is a valid default choice
        set_name = 'training'

    print('Loading FreiHAND dataset index ...')
    t = time.time()

    # assumed paths to data containers
    k_path = os.path.join(base_path, '%s_K.json' % set_name)
    mano_path = os.path.join(base_path, '%s_mano.json' % set_name)
    xyz_path = os.path.join(base_path, '%s_xyz.json' % set_name)

    # load if exist
    K_list = json_load(k_path)
    mano_list = json_load(mano_path)
    xyz_list = json_load(xyz_path)

    # should have all the same length
    assert len(K_list) == len(mano_list), 'Size mismatch.'
    assert len(K_list) == len(xyz_list), 'Size mismatch.'

    print('Loading of %d samples done in %.2f seconds' % (len(K_list), time.time()-t))
    return zip(K_list, mano_list, xyz_list)


def json_load(p):
    _assert_exist(p)
    with open(p, 'r') as fi:
        d = json.load(fi)
    return d


""" General util functions. """
def _assert_exist(p):
    msg = 'File does not exists: %s' % p
    assert os.path.exists(p), msg


def read_img_cv(idx, base_path, set_name, version=None):    # index, BASE_PATH(FH数据集路径), 'training', version
    if version is None:
        version = sample_version.gs

    if set_name == 'evaluation':
        assert version == sample_version.gs, 'This the only valid choice for samples from the evaluation split.'

    img_rgb_path = os.path.join(base_path, set_name, 'rgb', '%08d.jpg' % idx)
    _assert_exist(img_rgb_path)
    return img_rgb_path


class sample_version:
    gs = 'gs'  # green screen
    hom = 'hom'  # homogenized
    sample = 'sample'  # auto colorization with sample points
    auto = 'auto'  # auto colorization without sample points: automatic color hallucination

    db_size = db_size('training')

    @classmethod
    def valid_options(cls):
        return [cls.gs, cls.hom, cls.sample, cls.auto]

    @classmethod
    def check_valid(cls, version):
        msg = 'Invalid choice: "%s" (must be in %s)' % (version, cls.valid_options())
        assert version in cls.valid_options(), msg

    @classmethod
    def map_id(cls, id, version):
        cls.check_valid(version)
        return id + cls.db_size * cls.valid_options().index(version)


def projectPoints(xyz, K):
    """ Project 3D coordinates into image space. """
    xyz = np.array(xyz)
    K = np.array(K)
    uv = np.matmul(K, xyz.T).T
    return uv[:, :2] / uv[:, -1:]


if __name__ == "__main__":
    # 创建一个解析器
    parser = argparse.ArgumentParser()
    # 添加参数ImgDatPath
    parser.add_argument("--SaveDir", default=r"E:\Data\landmarks\FH\FH_from_whole_body_v2_6")
    parser.add_argument("--DataDir", default=r"E:\Data\landmarks\FreiHAND_pub_v2")
    # 解析参数
    args = parser.parse_args()
    main(args.DataDir, args.SaveDir)