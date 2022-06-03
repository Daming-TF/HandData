# 程序效果：
# 在线跑model记录5个末端关键点和手掌关键点坐标并与gt计算欧氏距离,当distance > threshold时输出图片对比窗口
# record信息保存在scripts/MarkRecord.txt文件
# 按键操作检测顺序
'''按键提醒：
     f：下一张
     b: 上一张
     m: 记录图片相对路径地址到本地目录（工作地址）的scripts/MarkRecord.txt文件'''

## 注意：(没有递归操作)
## 输入数据集格式如："../imgdate/YT3D/YT3D/images/train2017"

import cv2
import numpy as np
import time
from HandDet import HandDetModel  # provide a bound box of a hand（手掌检测）
from Hand2D import Hand2DModel  # predicting the hand skeleton（预测手势）
from tools import draw_2d_points, jsonshow
import argparse
import os

def main():
    hand_detector = HandDetModel(args.HandDetModelPATH)
    # onnx:一种开放的模型文件标准格式，那么我们将尝试把前面实现的多入多出三层神经网络保存为ONNX模型文件，以方便在不同的框架中都可以使用
    hand_2d_extractor = Hand2DModel(args.Hand2DModelPATH)

    filenames = os.listdir(args.ImgDatePath)
    # 返回指定文件夹包含的文件或文件夹名字列表
    img_firstpath = filenames[0]
    img_firstserial = int(os.path.splitext(img_firstpath)[0])
    img_lastpath = filenames[-1]
    img_lastserial = int(os.path.splitext(img_lastpath)[0])
    img_path = img_firstpath
    # 表示现在图片翻页的方向状态机 1表示向前
    flag = 1
    while 1:
        print(img_path)
        img_lonepath = os.path.join(args.ImgDatePath, img_path)
        img_name, img_ext = os.path.splitext(os.path.basename(img_path))
        img_bgr = cv2.imread(img_lonepath)
        # 色彩空间转化函数，图片格式转化为RGB格式,imread读取的是BGR格式
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        # 删除文件后缀
        img_serialnum = int(os.path.splitext(img_name)[0])

        # HandDetect Model
        start = time.time()
        # 返回定位目标框的两个坐标
        # 之所以hand_boxes是因为手的数量可能不止一个
        hand_boxes = hand_detector.detect(img_rgb)
        end = time.time()
        # 输出每帧目标框处理时间
        print('detector time: %f' % ((end - start) * 1000))

        # Hand2D Model
        start = time.time()
        hand_attrs = []
        # 把手势特征点记录到hand_attrs中
        for hand_box in hand_boxes:
            # 把截取的特定区域做姿势识别
            pts_2d, is_hand = hand_2d_extractor.predict(img_rgb, hand_box)
            if not is_hand:
                continue
            hand_attrs.append(pts_2d)
        end = time.time()
        # 输出每帧手势检测处理时间
        print('2d extractor time: %f' % ((end - start) * 1000))

        # Draw
        for hand_attr in hand_attrs:
            pts_2d = hand_attr
            # draw the hand landmark
            # bbox = coords_to_box(pts_2d)
            # cv2.rectangle(img_bgr, bbox[0], bbox[1], color=(0, 255, 0), thickness=3)
            img_bgr = draw_2d_points(pts_2d, img_bgr)

        # 计算模型输出和gt的距离
        img_gt, _,  gt_kp_points = jsonshow(img_path, args.ImgDatePath, args.JsonPath, 6)
        our_kp_points = pts_2d[[0,4,8,12,16,20],:].reshape(6,2)
        dx = (gt_kp_points - our_kp_points)[:,0]
        dy = (gt_kp_points - our_kp_points)[:,1]
        sp = img_rgb.shape
        dx = dx/sp[1]
        dy = dy/sp[0]
        res1 = np.sum(np.hypot(dx, dy))
        res2 = np.sum(np.abs(dx-dy), axis=0)
        print("欧式距离为："+str(res1))
        print("曼哈顿距离为："+str(res2))

        if res1 > 0.15 :
            canvas = np.hstack([img_bgr, img_gt])
            cv2.moveWindow("winname", 400, 300)
            cv2.imshow('Compare-left(our)-right(gt)', canvas)

            while 1:
                key = cv2.waitKey(0) & 0xFF
                if key == ord('f'):
                    flag = 1
                    img_serialnum = img_serialnum + 1
                    break;
                elif key == ord('b'):
                    flag = 0
                    img_serialnum = img_serialnum - 1
                    break;
                elif key == ord('m'):
                    # 如果filename不存在会自动创建， 'w' 表示写数据，写之前会清空文件中的原有数据！
                    with open(filename, 'a') as f:
                        f.write(args.ImgDatePath+'/'+img_path+"\n")
                        flag = 1
                        img_serialnum = img_serialnum + 1
                    print(str(img_serialnum)+img_ext+" is Unqualified")
                    print("This pic has been recorded in "+filename+"\n")
                    break;
                elif key == ord('q'):
                    return 1
        # img_name = '000000'+str(img_serialnum)+'.jpg'
        else:
            if flag == 1:
                img_serialnum = img_serialnum + 1
            elif flag == 0:
                img_serialnum = img_serialnum - 1

        if img_serialnum < img_firstserial or img_serialnum > img_lastserial:
            print(str(img_serialnum) + img_ext + ' does not exist')
            if img_serialnum < img_firstserial:
                img_serialnum = img_firstserial
            else:
                img_serialnum = img_lastserial
        img_path = str(img_serialnum).zfill(12) + img_ext

if __name__ == '__main__':
    # 创建一个解析器
    parser = argparse.ArgumentParser()
    parser.description = 'Please enter two parameters a and b ...'
    # 添加参数ImgDatPath
    parser.add_argument("--HandDetPath",
                        default=r"../library/models/models/HandDet.onnx",
                        help="this parameter is about the PATH of HandDet Model", dest="HandDetModelPATH", type=str)
    parser.add_argument("--Hand2DPath",
                        default=r"../library/models/models/model_best_2.onnx",
                        help="this parameter is about the PATH of Hand2D Model", dest="Hand2DModelPATH", type=str)
    parser.add_argument("--ImgDatePath",
                        default=r"../imgdate/YT3D/images/train2017",
                        help="this parameter is about the PATH of ImgDate", dest="ImgDatePath", type=str)
    '''parser.add_argument("--SavePath",
                        default=r"../Output",
                        help="this parameter is about the PATH of ImgSave", dest="SavePath", type=str)'''
    '''parser.add_argument("--RecordPath",
                        default=r"imgdate/Output",
                        help="this parameter is about the PATH of ImgSave", dest="RecordPath", type=str)'''
    parser.add_argument("--JsonPath",
                        default=r"../imgdate/YT3D/annotations/person_keypoints_train2017.json",
                        help="this parameter is about the PATH of JsonPath", dest="JsonPath", type=str)
    # 解析参数
    args = parser.parse_args()
    print("parameter 'hand_det_path' is :", args.HandDetModelPATH)
    print("parameter '2dpose_predict_path' is :", args.Hand2DModelPATH)
    print("parameter 'imgdate_path' is :", args.ImgDatePath)
    '''print("parameter 'imgsave_path' is :", args.SavePath)'''
    '''print("parameter 'recordinfo_path' is :", args.RecordPath)'''
    print("parameter 'jsoninfo_path' is :", args.JsonPath)
    # record信息（图片路径以及distance）保存路径
    filename = os.getcwd() + '\MarkRecord.txt'
    
    main()

