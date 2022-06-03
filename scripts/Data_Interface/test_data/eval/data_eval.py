from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
from collections import OrderedDict
import logging
import pylab,json
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
full_arch_name = 'pose_hrnet'

#  coco_results
def evaluate_predictions_on_coco(
        coco_gt, coco_dt, iou_type='keypoints'):
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
    plt.savefig(r'D:\My Documents\Desktop\毕设资料\竞品分析\PR_baidu.png')
    plt.show()


if __name__ == "__main__":
    gt_path = r"E:\test_data\test_data_from_whole_body\annotations\average_pseudo_labels_update-coco_id.json"   # 存放真实标签的路径             # r"G:\test_data\old_data\anno\person_keypoints_test2017.json"
    # 以list形式存储的检测结果字典，字典键值为{“image_id”,"category_id",“keypoints”,“score”}
    # Fix1_testdata_baidu_api_cocodt_format(score0.5).json
    # testdata_baidu_cocodt_format.json

    # Fix1_testdata_baidu_api_cocodt_format(score0.5).json
    # hand_test_02_v2-full.json
    # hand_test_02--mediapipe.json
    dt_path = r"E:\test_data\test_data_from_whole_body\annotations\mediapipe_full-vedio-coco_id.json"    # 存放检测结果的路径

    cocoGt = COCO(gt_path)
    cocoDt = COCO(dt_path)
    # coco_gt, json_result_file
    evaluate_predictions_on_coco(cocoGt, cocoDt)
