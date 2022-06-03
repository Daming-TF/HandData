import os

import cv2
import numpy as np

VIDEO_PATH = r"G:\test_data\hand_test_02.mp4"
newmodel_img_dir = r"G:\test_data\vedio_images\our"
baidu_img_dir = r"G:\test_data\vedio_images\baidu"
mediapipe_img_dir = r"G:\test_data\vedio_images\mediapipe_image"
SAVE_PATH = r"G:\test_data\vedio_images\comparision-all.mp4"

class VideoWriter(object):
    def __init__(self, save_name, cap, size=None):
        os.makedirs(os.path.dirname(save_name), exist_ok=True)

        # parameters of the video header
        # fps = cap.get(cv2.CAP_PROP_FPS)
        fps = 20
        frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.img_size = size if size is not None else frame_size
        codec = self.get_codec(save_name)

        self.writer = cv2.VideoWriter(save_name, codec, fps, self.img_size)

    @staticmethod
    def get_codec(save_name):
        _, ext = os.path.splitext(save_name)
        if ext == '.avi':
            codec = cv2.VideoWriter_fourcc(*'XVID')
        elif ext == '.mp4':
            codec = cv2.VideoWriter_fourcc(*'mp4v')
        else:
            raise Exception(" [!] Video extension of {} is not supported!".format(ext))

        return codec

    def write(self, frame):
        if frame.shape != self.img_size:
            frame = cv2.resize(frame, (self.img_size[0], self.img_size[1]), interpolation=cv2.INTER_CUBIC)
        self.writer.write(frame)

    def release(self):
        self.writer.release()

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    w, h = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    # 保存视频的函数，其作用是将录制的视频保存成filename
    writer = VideoWriter(SAVE_PATH, cap, (2 * w, h))

    filenames = os.listdir(newmodel_img_dir)
    for filename in filenames:
        our_img_path = os.path.join(newmodel_img_dir, filename)
        mediapipe_img_path = os.path.join(mediapipe_img_dir, filename)
        baidu_img_path = os.path.join(baidu_img_dir, filename)

        img1 = cv2.imread(our_img_path)
        img2 = cv2.imread(baidu_img_path)
        img3 = cv2.imread(mediapipe_img_path)
        img2 = cv2.flip(img2, 1)
        img3 = cv2.flip(img3, 1)
        canvas = np.hstack([img1, img2, img3])

        print(filename)
        cv2.imshow("left-our | middle-baidu | right-mediapipe", canvas)
        writer.write(canvas)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    writer.release()
    cap.release()

if __name__ == "__main__":
    main()