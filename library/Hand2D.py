import numpy as np
import cv2
import onnxruntime
from utils import box_to_center_scale, get_affine_transform, get_final_preds


class Hand2DModel(object):
    def __init__(self, model_path):
        """
        initialization
        :param model_path: path to onnx file
        """
        self.model = onnxruntime.InferenceSession(model_path)   #载入模型
        self.input_size = self.model.get_inputs()[0].shape[-2:]
        print('*'*50)
        print(f'self.input_size: {self.input_size}')
        print('*'*50)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def pre_process(self, img_rgb, box):
        #得到每个box的center和scale
        center, scale = box_to_center_scale(box, self.input_size[0], self.input_size[1])
        #针对每个person，获取原图到输入的仿射变换（平移、旋转、伸缩）矩阵
        trans = get_affine_transform(center, scale, 0, self.input_size)
        #仿射变换修正图片
        img_crop = cv2.warpAffine(img_rgb, trans, (int(self.input_size[0]), int(self.input_size[1])))
        #图片规范化
        img_norm = (img_crop.astype(np.float32) / 255. - self.mean) / self.std
        img_norm = img_norm.transpose(2, 0, 1)
        return img_norm, center, scale
        
    def post_process(self, model_outputs, center, scale):
        coords, confs = get_final_preds(model_outputs[0], [center], [scale])
        is_hand = np.sum(confs[0] > 0.3) >= 5
        return coords[0], is_hand
        
    def predict(self, img_rgb, box):
        """
        run forward with an image croped by boxes
        :param img_rgb: a numpy image [H, W, C] (RGB)
        :param box: [(x1, y1), (x2, y2)]
        :return: N x keypoints (21x2), is_hand flag
        """
        #得到仿射修正的规范格式图片以及每个box的center、scale
        img_norm, center, scale = self.pre_process(img_rgb, box)
        #模型输出
        model_outputs = self.model.run(None, {self.model.get_inputs()[0].name: [img_norm]}) #img_norm[None, ...]
        #返回特征点坐标
        pts_2d, is_hand = self.post_process(model_outputs, center, scale)
        return pts_2d, is_hand
        