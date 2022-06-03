# 程序功能：
# 通过输入存储数据集的路径，并根据每个数据包的文件结构，在Output中递归生成一样的文件结构存储结果
# 并在Output每个数据包的train、val、test路径下生成record.txt文件，用于存储每个Distan > Threshold的图片名字以及，欧氏距离值res1

## 注意：(有递归操作)
## 输入数据集格式如："../imgdate"
## 对比auto_recoddistance.py 这里调用的数据集中gt数据可能存在 x=0 && y=0 的点表示没检测到所以没有标的
## 所以在计算欧氏距离时输入21个点而并非六个点

from HandDet import HandDetModel  # provide a bound box of a hand（手掌检测）
from Hand2D import Hand2DModel  # predicting the hand skeleton（预测手势）
from tools import mkdir, draw_2d_points, coords_to_box, jsonshow, draw_text

import cv2
import time
import argparse
import os
import numpy as np

from tools import L2Distance_calculation_HFB

def main():
    check(args.ImgDatePath, args.SavePath)

def check(path, save_path):
    print(path)
    filenames = os.listdir(path)
    info_list = list()
    for filename in filenames:
        # a = os.path.isdir(os.path.join(path, filename))
        pts_2d = np.arange(21)
        if os.path.isdir(os.path.join(path, filename)) and filename in ["images", "annotations", "train2017", "val2017", "test2017"]:
            # b = os.path.join(path, filename)
            save_path1 = os.path.join(save_path, filename)
            print(filename)
            if not os.path.exists(save_path1):
                mkdir(save_path1, filename)
                path1 = os.path.join(path, filename)
                check(path1, save_path1)
        else:
            if filename.endswith('.json'):
                json_path = os.path.join(path, filename)
                json_list.append(json_path)


            elif filename.endswith('.jpg'):
                img_bgr = cv2.imread(os.path.join(path, filename))
                # 色彩空间转化函数，图片格式转化为RGB格式,imread读取的是BGR格式
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

                # start = time.time()
                # 返回定位目标框的两个坐标
                hand_boxes = hand_detector.detect(img_rgb)
                # end = time.time()
                # print('detector time: %f' % ((end - start) * 1000))

                # start = time.time()
                hand_attrs = []
                # 把手势特征点记录到hand_attrs中
                for hand_box in hand_boxes:
                    # 把截取的特定区域做姿势识别
                    pts_2d, is_hand = hand_2d_extractor.predict(img_rgb, hand_box)
                    if not is_hand:
                        continue
                    hand_attrs.append(pts_2d)
                end = time.time()
                # print('2d extractor time: %f' % ((end - start) * 1000))

                # draw
                for hand_attr in hand_attrs:
                    pts_2d = hand_attr
                    bbox = coords_to_box(pts_2d)
                    cv2.rectangle(img_bgr, bbox[0], bbox[1], color=(0, 255, 0), thickness=1)
                    img_bgr = draw_2d_points(pts_2d, img_bgr)

                keyword = path[path.rfind('\\')+1:]
                # print(json_list)
                for json_name in json_list:
                    if not json_name.find(keyword) == -1:
                        json_file = json_name
                # 返回gt图片以及21个关键点
                img_gt, img_txt, gt_kp_points = jsonshow(filename, path, json_file, 21)

                # our_kp_points = pts_2d
                print(filename)
                #res = L2Distance_calculation_YT3D(gt_kp_points, our_kp_points, img_rgb)
                res = L2Distance_calculation_HFB(gt_kp_points, pts_2d, img_rgb)



                if res > 0:
                    # print(os.path.join(save_path, img_file))
                    # cv2.imshow('debug', img_bgr)
                    if not os.path.exists(os.path.join(save_path, filename)):
                        img_gt = cv2.resize(img_gt, (700, 700), interpolation=cv2.INTER_LINEAR)
                        cv2.imwrite(os.path.join(save_path, filename), img_gt)

                        modul_output_path = os.path.join(save_path, "modul_output")
                        mkdir(modul_output_path, "modul_output")
                        img_bgr = cv2.resize(img_bgr, (700, 700), interpolation=cv2.INTER_LINEAR)
                        cv2.imwrite(os.path.join(modul_output_path, filename),img_bgr)

                        gt_txt_path = os.path.join(save_path, "gt_text")
                        mkdir(gt_txt_path, "gt_text")
                        img_txt = cv2.resize(img_txt, (700, 700), interpolation=cv2.INTER_LINEAR)
                        cv2.imwrite(os.path.join(gt_txt_path, filename), img_txt)


                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            return
                    info = filename + '\t' + str(res) + '\n'
                    info_list.append(info)
    with open((save_path + r'\\record.txt'), 'a') as f:
        for j in range(len(info_list)):
            f.write(info_list[j])
            #print(filename)
            #print("欧式距离为：" + str(res1) + '\n')


if __name__ == '__main__':
    # 创建一个解析器
    parser = argparse.ArgumentParser()
    parser.description = 'Please enter two parameters a and b ...'
    # 添加参数ImgDatPath
    parser.add_argument("--HandDetPath",
                        default=
                        r"D:\huya_AiBase\Project_hand\handpose\Hand2dDetection\library\models\models\HandDet.onnx",
                        help="this parameter is about the PATH of HandDet Model", dest="HandDetModelPATH", type=str)
    parser.add_argument("--Hand2DPath",
                        default=
                        r"D:\huya_AiBase\Project_hand\handpose\Hand2dDetection\library\models\models\model_best_2.onnx",
                        help="this parameter is about the PATH of Hand2D Model", dest="Hand2DModelPATH", type=str)
    parser.add_argument("--ImgDatePath",
                        default=r"G:\image\test",
                        help="this parameter is about the PATH of ImgDate", dest="ImgDatePath", type=str)
    parser.add_argument("--SavePath",
                        default=r"G:\Model_output",
                        help="this parameter is about the PATH of ImgSave", dest="SavePath", type=str)
    '''parser.add_argument("--RecordPath",
                        default=r"imgdate/Output",
                        help="this parameter is about the PATH of ImgSave", dest="RecordPath", type=str)'''
    # 解析参数
    args = parser.parse_args()
    print("parameter 'hand_det_path' is :", args.HandDetModelPATH)
    print("parameter '2dpose_predict_path' is :", args.Hand2DModelPATH)
    print("parameter 'imgdate_path' is :", args.ImgDatePath)
    print("parameter 'imgsave_path' is :", args.SavePath)
    '''print("parameter 'recordinfo_path' is :", args.RecordPath)'''
    '''print("parameter 'jsoninfo_path' is :", args.JsonPath)'''

    hand_detector = HandDetModel(args.HandDetModelPATH)
    # onnx:一种开放的模型文件标准格式，那么尝试把前面实现的多入多出三层神经网络保存为ONNX模型文件，以方便在不同的框架中都可以使用
    hand_2d_extractor = Hand2DModel(args.Hand2DModelPATH)
    # hand_2d_extractor = Hand2DModel('models/model_best_2.onnx')

    json_list = list()


    main()

