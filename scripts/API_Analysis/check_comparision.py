# 程序效果：
# 将auto_recoreddistance筛选出的“错误”图片中独帧显示
''' 可以通过按键操作
         下一张：'f'
         上一张：'b'
      标记并显示：'m' 记录不合格图片信息，流程如下
      '''
# 先将图片的路径信息保存在 Output/<数据包名>/<调用数据文件名>         例如：'../Output/YT3D/test2017.txt'
# 存储图片信息
## 同时将标记的图片保存在DatePath/record_date文件下                例如:'../Output/YT3D/YT3D/images/test2017/record_date'
## 输入数据格式如:"../Output/YT3D/YT3D/images/test2017"         例如:'YT3D/YT3D/images/test2017/000000300002.jpg'
## 需要修改图片信息记录文本的保存路径到/Output目录下的对应位置

## 相比find_fakewrong 2.0版本修改了成适配通用数据集（1.0版本需要输入图片以数字命名的）
## 其次对应mark_info 存储在../Output/<DATA NAME>路径下
## 输出对比变为：原图-gt输出-text输出

from tools import mkdir
import argparse
import os
import cv2
import numpy as np
# E:\OutputData_backup - 副本\YT3D\YT3D\images\train2017

SAVE_PATH = r"G:\test_data\vedio_images\badcase"
TXT_PATH = r"G:\test_data\vedio_images\badcase\badcase.txt"
TXT2_PATH = r"G:\test_data\vedio_images\badcase\no_det.txt"

def main():
    cv2.namedWindow("Show", cv2.WINDOW_NORMAL)
    filenames = os.listdir(args.DatePath)
    img_list = list()
    # 找出第一张图片并记录其图片序号
    for filename in filenames:
        if filename.endswith('.jpg'):
            img_list.append(filename)

    # 默认把图片信息保存在Output的下一个目录下，如调用默认数据路径"../Output/YT3D/YT3D/images/test2017"
    # mark图片信息保存路径为"../Output/YT3D/test2017.txt"

    i = 0
    while 1:
        if i == 0:
            j = 0
        else:
            j = i + 1
        img_path = os.path.join(args.DatePath, img_list[i])
        our_path = os.path.join(args.OurPath, img_list[j])
        baidu_path = os.path.join(args.BaiduPath, img_list[i])
        mediapipe_path = os.path.join(args.MediapipePath, img_list[i])

        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            # img = cv2.resize(img, (400, 400), interpolation=cv2.INTER_LINEAR)

            img_our = cv2.imread(our_path)
            img_our = cv2.flip(img_our, 1)
            img_baidu = cv2.imread(baidu_path)
            img_mediapipe = cv2.imread(mediapipe_path)
            # img_original = cv2.flip(img_original, 1)
            # img_original = cv2.resize(img_original, (400, 400), interpolation=cv2.INTER_LINEAR)

            canvas = np.hstack([img, img_our, img_baidu, img_mediapipe])
            cv2.imshow("show", canvas)
            # cv2.imshow('Smile when you see this', img)
            print(img_list[i])
            # cv2.imshow("SHOW", img)

            while 1:
                key = cv2.waitKey(0) & 0xFF
                if key == ord('f'):
                    i = i+1
                    break
                elif key == ord('b'):
                    i = i-1
                    break
                elif key == ord('m') or key == ord('p'):
                    # save_path = os.path.join(SAVE_PATH, img_list[i])
                    if key == ord('m'):
                        # 检查图片是否已经标记
                        img_examine_path = os.path.join(SAVE_PATH, img_list[i])
                        if not os.path.exists(img_examine_path):
                            cv2.imwrite(img_examine_path, canvas)
                            print(img_list[i] + "is recored")

                            with open(TXT_PATH, 'a') as f:
                                f.write(img_path + "\n")
                            print(img_list[i] + " is Unqualified")
                            print("This pic has been recorded in " + TXT_PATH + "\n")
                        else:
                            print(img_list[i] + "has beeen recored")

                    if key == ord('p'):
                        # 检查图片是否已经标记
                        img_examine_path = os.path.join(SAVE_PATH, img_list[i])
                        if not os.path.exists(img_examine_path):
                            print(img_list[i] + "is recored")

                            with open(TXT2_PATH, 'a') as f:
                                f.write(img_path + "\n")
                            print(img_list[i] + " is Unqualified")
                            print("This pic has been recorded in " + TXT_PATH + "\n")
                        else:
                            print(img_list[i] + "has beeen recored")

                    i = i + 1
                    break


                elif key == ord('q'):
                    return 1
                if i == len(img_list):
                    break

if __name__ == "__main__":
    # 创建一个解析器
    parser = argparse.ArgumentParser()
    parser.description = 'Please enter two parameters a and b ...'
    # 添加参数ImgDatPath
    # D:\huya_AiBase\Project_hand\handpose\Hand2dDetection\Output\HFB\HFB\images\test2017
    parser.add_argument("--DatePath",
                        default=r"G:\test_data\vedio_images\images",
                        help="this parameter is about the PATH of HandDet Model", dest="DatePath", type=str)
    parser.add_argument("--OurPath",
                        default=r"G:\test_data\vedio_images\our",
                        help="this parameter is about the PATH of HandDet Model", dest="OurPath", type=str)
    parser.add_argument("--BaiduPath",
                        default=r"G:\test_data\vedio_images\baidu",
                        help="this parameter is about the PATH of HandDet Model", dest="BaiduPath", type=str)
    parser.add_argument("--MediapipePath",
                        default=r"G:\test_data\vedio_images\mediapipe_image",
                        help="this parameter is about the PATH of HandDet Model", dest="MediapipePath", type=str)
    args = parser.parse_args()
    print("parameter 'DatePath' is :", args.DatePath)
    print("parameter 'OriginalPath' is :", args.OurPath)
    print("parameter 'DatePath' is :", args.BaiduPath)
    print("parameter 'OriginalPath' is :", args.MediapipePath)
    '''print("parameter 'MOutputPath' is :", args.MOutputPath)'''
    main()

