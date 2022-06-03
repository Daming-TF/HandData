import os
import json
import argparse
import update_tools


def set_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--BatchSamplePath", help="批次数据的json文件路径",
            default=r"E:\数据标记反馈\new_test_data(6945&7035)\批次样本")
    parser.add_argument("--InvalidTxtDir", help="无效数据的txt文件路径,手动筛选的记录文件",
            default=r"E:\test_data\test_data_from_whole_body\record\badcase_invalid.txt")
    parser.add_argument("--InvalidJsonPath", help="无效数据的json文件路径",
            default=r"E:\数据标记反馈\new_test_data(6945&7035)\无效样本")
    parser.add_argument("--JsonPath", help="数据集json文件路径——cocoJson文件路径",
            default=rf"E:\test_data\test_data_from_whole_body\annotations\coco_eval\gt")
    parser.add_argument("--JsonSavePath", help="更新后文件写入路径",
                        default=rf"E:\test_data\test_data_from_whole_body\annotations\coco_eval\gt")       # person_keypoints_{mode}-update.json
    parser.add_argument("--Debug", help="更新后是否保存json文件",
                        default=0)
    args = parser.parse_args()
    return args


def main():
    vedio_names = ["hand_test_01", "hand_test_02", "hand_test_03", "hand_test_04", "hand_test_05", "hand_test_06",
                  "hand_test_07", "hand_test_08", "hand_test_09", "hand_test_10"]
    args = set_parser()
    invalid_txt_dir = args.InvalidTxtDir
    invalid_json_path = args.InvalidJsonPath
    batch_sample_path = args.BatchSamplePath
    json_path = args.JsonPath
    debug = args.Debug

    for vedio_name in vedio_names:
        save_path = args.JsonSavePath
        json_dir = os.path.join(json_path, vedio_name, vedio_name+'-gt.json')
        with open(json_dir, 'r')as f:
            json_data = json.load(f)

        save_path = os.path.join(save_path, vedio_name)

        print(f"Cleaning from batch sample json file......")
        save_dir = os.path.join(save_path, f'{vedio_name}_update_with_beach_sample.json')
        update_tools.update_from_batch_json(json_data, batch_sample_path, save_dir, debug)

        print(f"Cleaning from invalid txt file......")
        save_dir = os.path.join(save_path, f'{vedio_name}_wo_invalid_txt.json')
        update_tools.update_from_invalid_txt(json_data, invalid_txt_dir, save_dir, debug)

        print(f"Cleaning from invalid sample json file......")
        save_dir = os.path.join(save_path, f'{vedio_name}_update.json')
        update_tools.update_from_invalid_json(json_data, invalid_json_path, save_dir)


if __name__ == '__main__':
    main()