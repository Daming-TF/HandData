import os
import argparse
import update_tools

from json_tools import load_json_data


def set_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--BatchSampleDir", help="批次数据的json文件路径",
            default=r"E:\数据标记反馈\hxg(6143)\批次样本")
    # parser.add_argument("--InvalidTxtDir", help="无效数据的txt文件路径,手动筛选的记录文件",
    #         default=r"E:\test_data\test_data_from_whole_body\record\badcase_invalid.txt")
    parser.add_argument("--InvalidJsonDir", help="无效数据的json文件路径",
            default=r"E:\数据标记反馈\hxg(6143)\无效样本")
    parser.add_argument("--JsonPath", help="数据集json文件路径——cocoJson文件路径",
            default=rf"E:\Data\landmarks\handpose_x_gesture_v1\HXG_from_whole_body_v2_6\annotations\person_keypoints_train2017.json")
    parser.add_argument("--JsonSaveDir", help="更新后文件写入路径",
                        default=rf"E:\Data\landmarks\handpose_x_gesture_v1\HXG_from_whole_body_v2_6\annotations")
    parser.add_argument("--Debug", help="更新后是否保存json文件",
                        default=1)
    args = parser.parse_args()
    return args


def main():
    args = set_parser()
    invalid_json_dir = args.InvalidJsonDir
    batch_sample_dir = args.BatchSampleDir
    json_path = args.JsonPath
    json_save_dir = args.JsonSaveDir
    debug = args.Debug

    images_dict, annotations_dict = load_json_data(json_path)

    save_name = os.path.basename(json_path).split('.json')[0]

    print(f"Covering from batch sample json file......")
    save_path = os.path.join(json_save_dir, f'{save_name}_update_with_beach_sample.json')
    update_tools.update_from_batch_json(images_dict, annotations_dict, batch_sample_dir, save_path, debug)

    print(f"Culling from invalid sample json file......")
    save_path = os.path.join(json_save_dir, f'{save_name}_update.json')
    update_tools.update_from_invalid_json(images_dict, annotations_dict, invalid_json_dir, save_path)


if __name__ == '__main__':
    main()