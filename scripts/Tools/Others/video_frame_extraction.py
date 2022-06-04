# -*- coding:utf8 -*-
"""
功能： 视频抽帧保存到指定地址
"""

import time
import cv2
import os
import shutil


def get_frame_from_video(save_dir, video_dir, interval, start_id):
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
    # else:
    #     shutil.rmtree(save_dir)
    #     os.makedirs(save_dir)
    #     print('path of %s already exist and rebuild' % save_dir)

    # 开始读视频
    video_capture = cv2.VideoCapture(video_dir)
    i = 0
    image_id = start_id

    while True:
        success, frame = video_capture.read()
        if not success:
            print('video is all read')
            break
        i += 1
        time.sleep(0.001)
        if i % interval == 0:
            # 保存图片
            save_name = str(image_id).zfill(10) + '.jpg'
            image_id += 1
            cv2.imshow("show", frame)
            cv2.imwrite(os.path.join(save_dir, save_name), frame)
            # print('image of %s is saved' % save_name)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print(f"There are {int(image_id)-start_id} frames in {os.path.basename(vedio_dir)}")
    return int(image_id)


if __name__ == '__main__':
    name = 'zhiwen'
    # 视频文件名字
    video_path = fr'G:\test_data\hardcase_data\vedio\{name}\{name}'
    save_dir = fr'G:\test_data\hardcase_data\dataset\{name}\no_frame_images'
    interval = 1
    start_id = 0

    vedio_names = os.listdir(video_path)

    cv2.namedWindow("show", cv2.WINDOW_NORMAL)

    for i in range(len(vedio_names)):
        vedio_name = vedio_names[i]
        vedio_dir = os.path.join(video_path, vedio_name)

        print(f"Now start drawing frames on {vedio_dir}")
        start_id = get_frame_from_video(save_dir, vedio_dir, interval, start_id)
