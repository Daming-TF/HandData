# -*- coding:utf8 -*-
import time

import cv2
import os
import shutil

def get_frame_from_video(save_dir, video_dir, interval):
    """
    Args:
        video_name:输入视频名字
        interval: 保存图片的帧率间隔
    Returns:
    """

    # 保存图片的路径
    filepath, _ = os.path.split(video_dir)
    # save_path = os.path.join(filepath, "vedio_images/images")
    is_exists = os.path.exists(save_dir)
    if not is_exists:
        os.makedirs(save_dir)
        print('path of %s is build' % save_dir)
    else:
        shutil.rmtree(save_dir)
        os.makedirs(save_dir)
        print('path of %s already exist and rebuild' % save_dir)

    # 开始读视频
    video_capture = cv2.VideoCapture(video_dir)
    i = 0
    j = 0

    while True:
        success, frame = video_capture.read()
        i += 1
        time.sleep(0.001)
        if i % interval == 0:
            # 保存图片
            save_name = str(j).zfill(10) + '.jpg'
            j += 1
            cv2.imshow("show", frame)
            cv2.imwrite(os.path.join(save_dir, save_name), frame)
            print('image of %s is saved' % save_name)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if not success:
            print('video is all read')
            break


if __name__ == '__main__':
    # 视频文件名字
    video_dir = r'G:\test_data\hardcase_data\vedio\aiyu\aiyu\aiyu_03.MP4'
    save_dir = r'E:\test_data\test_video\annotations\hand_test_01\test'
    interval = 1
    get_frame_from_video(save_dir, video_dir, interval)
