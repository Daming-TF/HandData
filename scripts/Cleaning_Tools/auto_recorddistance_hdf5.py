# 程序功能：
# 通过输入存储数据集的路径，并根据每个数据包的文件结构，在Output中递归生成一样的文件结构存储结果
# 并在Output每个数据包的train、val、test路径下生成record.txt文件，用于存储每个Distan > Threshold的图片名字以及，欧氏距离值res1

## 注意：(有递归操作)
## 输入数据集格式如："../imgdate"

from HandDet import HandDetModel  # provide a bound box of a hand（手掌检测）
from Hand2D import Hand2DModel  # predicting the hand skeleton（预测手势）
from tools import mkdir, draw_2d_points, coords_to_box, jsonshow, store_many_hdf5
import cv2
import time
import argparse
import os
import numpy as np

def main():
    check(args.ImgDatePath, args.SavePath)

def check(path, save_path):
    # print(path)
    filenames = os.listdir(path)
    # 存储错误信息：文件名+距离
    info_list = list()
    # 存储错误图片的列表(N , 400, 400, 3)
    images = []
    for filename in filenames:
        # a = os.path.isdir(os.path.join(path, filename))
        if os.path.isdir(os.path.join(path, filename)):
            # b = os.path.join(path, filename)
            save_path1 = os.path.join(save_path, filename)
            print(filename)
            if not os.path.exists(save_path1):
                mkdir(save_path1, filename)
                path1 = os.path.join(path, filename)
                check(path1, save_path1)
            if not len(images):
                test_image_path = os.path.join(args.SavePath, "test.jpg")
                test_image = cv2.imread(test_image_path)
                test_image = cv2.resize(test_image, (400, 400), interpolation=cv2.INTER_LINEAR)
                images.append(test_image)
        else:
            if filename.endswith('.json'):
                json_path = os.path.join(path, filename)
                json_list.append(json_path)
                # 防止输入到生成hdf5文件函数的images为空，导致images[0]报错
                if not len(images):
                    test_image_path = os.path.join(args.SavePath, "test.jpg")
                    test_image = cv2.imread(test_image_path)
                    test_image = cv2.resize(test_image, (400, 400), interpolation=cv2.INTER_LINEAR)
                    images.append(test_image)


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
                    cv2.rectangle(img_bgr, bbox[0], bbox[1], color=(0, 255, 0), thickness=3)
                    img_bgr = draw_2d_points(pts_2d, img_bgr)

                keyword = path[path.rfind('\\')+1:]
                # print(json_list)
                for json_name in json_list:
                    if not json_name.find(keyword) == -1:
                        json_file = json_name
                img_gt, gt_kp_points = jsonshow(filename, path, json_file)
                our_kp_points = pts_2d[[0, 4, 8, 12, 16, 20], :]
                dx = (gt_kp_points - our_kp_points)[:, 0]
                dy = (gt_kp_points - our_kp_points)[:, 1]
                sp = img_rgb.shape
                dx = dx / sp[1]
                dy = dy / sp[0]
                res1 = np.sum(np.hypot(dx, dy))
                # res2 = np.sum(np.abs(dx - dy), axis=0)
                # print("欧式距离为：" + str(res1))
                # print("曼哈顿距离为：" + str(res2))

                if res1 > 0.15:
                    # print(os.path.join(save_path, img_file))
                    # cv2.imshow('debug', img_bgr)
                    '''if not os.path.exists(os.path.join(save_path, filename)):
                        cv2.imwrite(os.path.join(save_path, filename), img_gt)
                        # print(os.path.join(save_path, filename))
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            return
                    info = filename + '\t' + str(res1) + '\n'
                    info_list.append(info)'''
                    image = cv2.resize(img_rgb, (400, 400), interpolation=cv2.INTER_LINEAR)
                    images.append(image)
            else:
                if not len(images):
                    test_image_path = os.path.join(args.SavePath, "test.jpg")
                    test_image = cv2.imread(test_image_path)
                    test_image = cv2.resize(test_image, (400, 400), interpolation=cv2.INTER_LINEAR)
                    images.append(test_image)



    store_many_hdf5(images, save_path)
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
                        default=r"../library/models/HandDet.onnx",
                        help="this parameter is about the PATH of HandDet Model", dest="HandDetModelPATH", type=str)
    parser.add_argument("--Hand2DPath",
                        default=r"../library/models/model_best_2.onnx",
                        help="this parameter is about the PATH of Hand2D Model", dest="Hand2DModelPATH", type=str)
    parser.add_argument("--ImgDatePath",
                        default=r"../imgdate",
                        help="this parameter is about the PATH of ImgDate", dest="ImgDatePath", type=str)
    parser.add_argument("--SavePath",
                        default=r"D:../Output",
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

