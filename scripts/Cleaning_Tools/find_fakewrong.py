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
def main():
    filenames = os.listdir(args.DatePath)
    record_path = os.path.join(args.DatePath, 'record_date')
    # 生成后面mark图片的存储文件夹
    mkdir(record_path, "record_date")
    img_list = list()
    # 找出第一张图片并记录其图片序号
    for filename in filenames:
        if filename.endswith('.jpg'):
            img_list.append(filename)

    # 默认把图片信息保存在Output的下一个目录下，如调用默认数据路径"../Output/YT3D/YT3D/images/test2017"
    # mark图片信息保存路径为"../Output/YT3D/test2017.txt"
    num = args.DatePath.index('Output')
    num1 = args.DatePath.index('\\', num)
    num2 = args.DatePath.index('\\', num1+1)

    Data_filename = args.DatePath[num+7: args.DatePath.index('\\', num+7)]
    txtfile_name = f"{Data_filename}-" + args.DatePath[args.DatePath.rfind('\\') + 1:] + '.txt'

    txtfile_path = os.path.join(args.DatePath[:num2], txtfile_name)
    record_path_info = args.DatePath[num1+1:]

    print(txtfile_path)
    i = 0
    while 1:
        img_path = os.path.join(args.DatePath, img_list[i])
        original_path = os.path.join(args.OriginalPath, img_list[i])
        gt_text_path = os.path.join(args.DatePath, "gt_text", img_list[i])
        img_model_path = os.path.join(args.DatePath, "modul_output", img_list[i])

        if os.path.exists(img_path):

            img = cv2.imread(img_path)
            img = cv2.resize(img, (400, 400), interpolation=cv2.INTER_LINEAR)

            img_original = cv2.imread(original_path)
            img_original = cv2.resize(img_original, (400, 400), interpolation=cv2.INTER_LINEAR)

            img_gt_text = cv2.imread(gt_text_path)
            img_gt_text = cv2.resize(img_gt_text, (400, 400), interpolation=cv2.INTER_LINEAR)

            img_model = cv2.imread(img_model_path)
            img_model = cv2.resize(img_model, (400, 400), interpolation=cv2.INTER_LINEAR)


            canvas1 = np.hstack([ img_original, img_model])
            canvas2 = np.hstack([img_gt_text, img])
            canvas3 = np.vstack([canvas1,canvas2])
            cv2.moveWindow("winname", 400, 300)
            cv2.imshow("Compare-'model'-'original'-'gttext'-'gt'", canvas3)
            # cv2.imshow('Smile when you see this', img)

            while 1:
                print(img_list[i])
                key = cv2.waitKey(0) & 0xFF
                if key == ord('f'):
                    i = i+1
                    break
                elif key == ord('b'):
                    i = i-1
                    break
                elif key == ord('m'):
                    record_info = os.path.join('.',record_path_info, img_list[i])

                    # 检查图片是否已经标记
                    img_examine_path = os.path.join(record_path, img_list[i])
                    if not os.path.exists(img_examine_path):
                        cv2.imwrite(img_examine_path, img)
                        print(img_list[i] + "is recored")

                        with open(txtfile_path, 'a') as f:
                            f.write(record_info + "\n")
                        print(img_list[i]+ " is Unqualified")
                        print("This pic has been recorded in " + img_examine_path + "\n")
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
                        default=r"G:\test_data\vedio_images\baidu",
                        help="this parameter is about the PATH of HandDet Model", dest="DatePath", type=str)
    parser.add_argument("--OriginalPath",
                        default=r"G:\test_data\vedio_images\our",
                        help="this parameter is about the PATH of HandDet Model", dest="OriginalPath", type=str)
    '''parser.add_argument("--MOutputPath",
                        default=r"F:\Model_output\Output\hand143_panopticdb\hand143_panopticdb\images\test2017\modul_output",
                        help="this parameter is about the PATH of HandDet Model", dest="MOutputPath", type=str)'''
    '''parser.add_argument("--JsonPath",
                        default=r"../imgdate/YT3D/YT3D/annotations/person_keypoints_test2017",
                        help="this parameter is about the PATH of HandDet Model", dest="json_path", type=str)'''
    args = parser.parse_args()
    print("parameter 'DatePath' is :", args.DatePath)
    print("parameter 'OriginalPath' is :", args.OriginalPath)
    '''print("parameter 'MOutputPath' is :", args.MOutputPath)'''
    main()

