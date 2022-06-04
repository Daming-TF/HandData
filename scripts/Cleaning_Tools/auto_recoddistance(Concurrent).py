# 程序功能：
# 通过输入存储数据集的路径，并根据每个数据包的文件结构，在Output中递归生成一样的文件结构存储结果
# 根据设置的process_num，把每个文件数据列表均分成process_num份，实现并发加速
# 并在Output每个数据包的train、val、test路径下生成record.txt文件，用于存储每个Distan > Threshold的图片名字以及，欧氏距离值res1

import cv2
import time
import argparse
import os
import numpy as np
from multiprocessing import Process, Queue, Pool, cpu_count

# from tools import L2Distance_calculation_HFB
from tqdm import tqdm
import copy

from library.HandDet import HandDetModel  # provide a bound box of a hand（手掌检测）
from library.Hand2D import Hand2DModel  # predicting the hand skeleton（预测手势）
from library.tools import mkdir, draw_2d_points, coords_to_box, jsonshow, draw_text, MAX_Divide_Min_Distance_calculation


json_list = list()
def main():
    save_path = os.path.join(args.SavePath, os.path.basename(args.ImgDatePath))
    check(args.ImgDatePath, save_path)


def check(path, save_path):
    # path 为当前搜索路径
    print(path)
    filenames = os.listdir(path)
    image_paths = list()

    for filename in filenames :
        # a = os.path.isdir(os.path.join(path, filename))
        pts_2d = np.arange(42).reshape(21,2)

        if os.path.isdir(os.path.join(path, filename)):
            if not filename in ["images", "annotations", "test2017", "train2017", "val2017"]:
                continue
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
                image_path = os.path.join(path, filename)
                image_paths.append(image_path)

    json_keyword = os.path.basename(path)
    json_file = ''
    for json_name in json_list:
        if not json_name.find(json_keyword) == -1:
            json_file = json_name

    if not len(image_paths) == 0:

        image_path_len = len(image_paths)
        demarcation = 0
        block_len = image_path_len // process_num
        image_blocks = list()
        for i in range(process_num):
            if not i == process_num-1:
                image_block = image_paths[demarcation : demarcation+block_len]
                image_blocks.append(image_block)
                demarcation = demarcation + block_len
                a = len(image_block)
            else:
                image_block = image_paths[demarcation:]
                image_blocks.append(image_block)

        # 父进程创建Queue队列，并传给各个子进程：
        num_len = 20000
        queue_info = Queue(num_len)
        process_list = [Process(target=read_image,
                                args=(queue_info, image_blocks[i], save_path,
                                      args.HandDetModelPATH, args.Hand2DModelPATH, json_file))
                        for i in range(process_num)]
        for i in range(process_num):
            process_list[i].start()

        # 当前位置阻塞主进程，带执行join()的进程结束后再继续执行主进程
        for i in range(process_num):
            process_list[i].join()

        # with open((save_path + r'\\record.txt'), 'a') as f:
        #     while not queue_info.empty():
        #         # get函数中的参数True，表示最多阻塞 timeout 秒，如果在这段时间内项目不能得到，将引发 Empty 异常
        #         info = queue_info.get(True)
        #         f.write(info)


def read_image(queue_info, image_block, save_path, handdet_modelpath, hand2d_modelpath, json_file):
    # 初始化设置
    # 设置图片当前搜索路径以及json文件名
    path = os.path.dirname(image_block[0])

    is_exist = os.path.exists(save_path)
    if not is_exist:
        os.mkdir(save_path)
    # 设置模型输出图片和打印gt数字图片的存储路径
    modul_output_path = os.path.join(save_path, "modul_output")
    mkdir(modul_output_path, "modul_output")
    gt_txt_path = os.path.join(save_path, "gt_text")
    mkdir(gt_txt_path, "gt_text")

    hand_detector = HandDetModel(handdet_modelpath)
    # onnx:一种开放的模型文件标准格式，那么尝试把前面实现的多入多出三层神经网络保存为ONNX模型文件，以方便在不同的框架中都可以使用
    hand_2d_extractor = Hand2DModel(hand2d_modelpath)
    # hand_2d_extractor = Hand2DModel('models/model_best_2.onnx')

    pts_2d = np.arange(42).reshape(21, 2)
    info_list = list()
    # 遍历图片进行欧氏距离计算
    for image_path in tqdm(image_block):
        img_bgr = cv2.imread(image_path)
        # 色彩空间转化函数，图片格式转化为RGB格式,imread读取的是BGR格式
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_modeloutput = copy.deepcopy(img_bgr)

        # start = time.time()
        # 返回定位目标框的两个坐标
        hand_boxes = hand_detector.detect(img_rgb)
        # end = time.time()
        # print('detector time: %f' % ((end - start) * 1000))

        # start = time.time()
        hand_attrs = []
        # 把手势特征点记录到hand_attrs中
        for hand_box in hand_boxes:
            # 把截取的特定区域做姿势识别，pts_2d表示21*2的关键点数组
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

            img_modeloutput = draw_2d_points(pts_2d, img_bgr)
            cv2.rectangle(img_modeloutput, bbox[0], bbox[1], color=(0, 255, 0), thickness=1)


        filename = os.path.basename(image_path)
        # 返回gt图片以及21个关键点
        # img_gt 表示打印gt骨骼的图片
        # img_text 表示打印关键点数字的图片
        # gt_kp_points 表示n*2的关键点坐标数组，n表示输入jsonshow函数的最后一个参数
        img_gt, img_text, gt_kp_points = jsonshow(filename, path, json_file, 21)

        # our_kp_points = pts_2d
        # print(filename)
        # res = L2Distance_calculation_YT3D(gt_kp_points, our_kp_points, img_rgb)
        res = MAX_Divide_Min_Distance_calculation(gt_kp_points, pts_2d, img_bgr)

        if res > 0:
            # print(os.path.join(save_path, img_file))
            # cv2.imshow('debug', img_bgr)
            if not os.path.exists(os.path.join(save_path, filename)):
                img_gt = cv2.resize(img_gt, (700, 700), interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(os.path.join(save_path, filename), img_gt)

                modul_output_path = os.path.join(save_path, "modul_output")
                mkdir(modul_output_path, "modul_output")
                img_modeloutput = cv2.resize(img_modeloutput, (700, 700), interpolation=cv2.INTER_LINEAR)
                location = [int(img_modeloutput.shape[0] / 10), int(img_modeloutput.shape[1] * 9 / 10)]
                cv2.putText(img_modeloutput, f"{res}", (location[0], location[1]), cv2.FONT_HERSHEY_COMPLEX, 1.0,
                            (37, 50, 232), 2)
                cv2.imwrite(os.path.join(modul_output_path, filename), img_modeloutput)

                gt_txt_path = os.path.join(save_path, "gt_text")
                mkdir(gt_txt_path, "gt_text")
                img_text = cv2.resize(img_text, (700, 700), interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(os.path.join(gt_txt_path, filename), img_text)

                # badpoint_text_path = os.path.join(save_path, "badpoint_text")
                # mkdir(badpoint_text_path, "badpoint_text")
                # img_badpoint = cv2.resize(img_badpoint, (700, 700), interpolation=cv2.INTER_LINEAR)
                # cv2.imwrite(os.path.join(badpoint_text_path, filename), img_badpoint)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    return
            # info_list = filename + '\t' + str(res) + '\n'
            # queue_info.put(info)
            info_list.append(filename + '\t' + str(res) + '\n')
    with open((save_path + r'\\record.txt'), 'a') as f:
        for j in range(len(info_list)):
            f.write(info_list[j])


if __name__ == '__main__':
    # 创建一个解析器
    parser = argparse.ArgumentParser()
    # parser.description = 'Please enter two parameters a and b ...'
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
                        default=r"E:\L2_eval\new_data",
                        help="this parameter is about the PATH of ImgDate", dest="ImgDatePath", type=str)
    parser.add_argument("--SavePath",
                        default=r"E:\L2_eval\new_data\output\index5-2-3",
                        help="this parameter is about the PATH of ImgSave", dest="SavePath", type=str)
    # parser.add_argument("--RecordPath",
    #                     default=r"imgdate/Output",
    #                     help="this parameter is about the PATH of ImgSave", dest="RecordPath", type=str)

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

    process_num = 6     # 进程数
    main()

