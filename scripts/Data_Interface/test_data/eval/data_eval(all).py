import os.path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
from collections import OrderedDict
import logging
import pylab,json
import matplotlib.pyplot as plt

gt_suffix = '-gt.json'
dt_suffix = '-mediapipe-full-id_start_from_1.json'     # _v2-full.json, --mediapipe.json, --mediapipe-lite.json, _v3-base.json, _v3-align.json


#  coco_results
def evaluate_predictions_on_coco(iou_type='keypoints'):
    vedio_names = ["hand_test_01", "hand_test_02", "hand_test_03", "hand_test_04", "hand_test_05", "hand_test_06",
                   "hand_test_07", "hand_test_08", "hand_test_09", "hand_test_10"]
    # vedio_names = ["hand_test_01", "hand_test_03", "hand_test_04", "hand_test_09", "hand_test_10"]
    json_path = r'E:\test_data\test_data_from_whole_body\annotations\coco_eval'

    for vedio_name in vedio_names:
        print(f"""
        
————————————————Evaluating the vedio about >>{vedio_name}<< ({dt_suffix.split('.')[0]})————————————————————————
        
        """)
        gt_path = os.path.join(json_path, 'gt', vedio_name, vedio_name+gt_suffix)
        dt_path = os.path.join(json_path, 'dt', vedio_name, vedio_name+dt_suffix)

        coco_gt = COCO(gt_path)
        coco_dt = COCO(dt_path)

        coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
        coco_eval.params.useSegm = None
        kpt_oks_sigmas = np.ones(21) * 0.35 / 10.0
        # np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62,.62, 1.07, 1.07, .87, .87, .89, .89])/10.0
        coco_eval.params.kpt_oks_sigmas = kpt_oks_sigmas

        coco_eval.evaluate()    # 得到单张图片在特定类别，特定面积阈值内，特定最大检测数下的所有阈值检测结果
        coco_eval.accumulate()      # 对这些单张图片的结果进行积累计算
        coco_eval.summarize()       # 根据传入IoU阈值、面积阈值、最大检测数这些参数返回对应的mAp与mAR

        pr_arrays = []

        pr_array1 = coco_eval.eval['precision'][0, :, 0, 0, 0]
        pr_array2 = coco_eval.eval['precision'][5, :, 0, 0, 0]
        pr_array3 = coco_eval.eval['precision'][9, :, 0, 0, 0]
        x = np.arange(0.0, 1.01, 0.01)

        plt.title('OUR MODULE')
        plt.xlabel('recall')
        plt.ylabel('precision')
        # 调整x,y坐标范围
        plt.xlim(0, 1.0)
        plt.ylim(0, 1.01)
        # 显示背景的网格线
        plt.grid(True)

        # plot([x], y, [fmt], data=None, **kwargs)
        # 可选参数[fmt] = '[color][marker][line]'
        # 是一个字符串来定义图的基本属性如：颜色（color），点型（marker），线型（linestyle）,具体形式
        plt.plot(x, pr_array1, 'b-', label='IoU=0.5')
        plt.plot(x, pr_array2, 'c-', label='IoU=0.75')
        plt.plot(x, pr_array3, 'y-', label='IoU=0.95')

        # plt.legend()函数的作用是给图像加图例
        plt.legend(loc="lower right")
        # plt.savefig(r'D:\My Documents\Desktop\毕设资料\竞品分析\PR_baidu.png')
        plt.show()


if __name__ == "__main__":
    evaluate_predictions_on_coco()
