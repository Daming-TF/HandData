# 工具包
import os
import sys
import cv2
import json
import numpy as np



pro_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(pro_dir)

# 读取json文件并返回gt图片以及6个关键点的数组
def jsonshow(img_name, img_folder, json_path, n):
    with open(json_path, "r") as f:
        data = json.load(f)
        # 该列表每个元素皆为一个img信息数据字典
        # 包括'license', 'file_name', 'coco_url', 'height', 'width', 'data_captured', 'flickr_url', 'id'
        imgs_info = data['images']
        # annos_info数据包括'segmentation', 'num_keypoints', 'image_id', 'bbox', 'category_id', 'id',
        annos_info = data['annotations']

    num_imgs = len(imgs_info)
    # 生成0-num_imgs数字列表
    for i in range(num_imgs):
        img_info = imgs_info[i]
        anno_info = annos_info[i]
        # img_info['id'] ！= anno_info['id']情况触发异常
        assert img_info['id'] == anno_info['id']

        if img_info['file_name'] == img_name:
            img_path = os.path.join(img_folder, img_info['file_name'])
            # 转化的原因：列表创建指针数=元素数  增加内存和cpu消耗，对于存数字操作一般转化为数组
            # 将输入转换为数组并以21行3列形式展开，第三列数据表示图片状态（遮挡/正常）
            kp_points = np.asarray(anno_info['keypoints']).reshape((21, 3))
            # bbox是四个坐标数据，如第一帧图片输出数据为(84, 84, 362, 362)
            bbox = anno_info['bbox']
            x, y, h, w = bbox

            img = cv2.imread(img_path)
            img_point = draw_2d_points(kp_points, img)
            img_text = draw_text(kp_points, img)
            # 在图像上绘制一个简单的矩形
            #img = cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 3)
            if n == 6:
                sixpoints = kp_points[[0, 4, 8, 12, 16, 20], 0:2]
                return img_point, img_text, sixpoints
                # cv2.imshow("GT", img)
                # if cv2.waitKey(0) == 27:
                #     exec("Esc clicked!")
            elif n == 21:
                return img_point, img_text, kp_points[:, 0:2]

# 检查文件是否存在并生成
def mkdir(path, name):
    folder = os.path.exists(path)
    # 判断是否存在文件夹如果不存在则创建为文件夹
    if not folder:
        # makedirs 创建文件时如果路径不存在会创建这个路径
        os.makedirs(path)
        print( "--- make new file"+"<"+name+">"+" ---")
    else:
        return 0

# 检查文件是否存在，若存在则加后缀递归生成
def mkdir_avoid_overwrit(filename):
    n=[0]
    def check_meta(file_name):
        file_name_new=file_name
        if os.path.isfile(file_name):
            file_name_new=file_name[:file_name.rfind('.')]+'_'+str(n[0])+file_name[file_name.rfind('.'):]
            n[0]+=1
        if os.path.isfile(file_name_new):
            file_name_new=check_meta(file_name)
        return file_name_new
    return_name = check_meta(filename)
    return return_name


line_colour = [(0, 0, 255), (0, 255, 255), (0, 255, 0), (255, 255, 0), (255, 0, 0)]

# 输出2d特征点
def draw_2d_points(points, im_ori, finger_num = 21, line_thickness=2, circle_size=4):
    '''

    Parameters
    ----------
    points: An array that satisfies the 21*3 format
    im_ori: oright image
    finger_num: 标识输出landmarks模式 (landmarks数量)

    Returns
    -------
    im: a image with landmarks
    '''
    if finger_num == 0:
        raise SystemExit('the param "finger_num" is not allowed to set to "0"')
    im = im_ori.copy()
    NUM_JOINTS = points.shape[0]
    for i in range(NUM_JOINTS):
        point = points[i]
        x = int(point[0])
        y = int(point[1])

        if i == 0:
            # 记录手腕关键点
            # rootx,rooty表示手腕坐标
            rootx = x
            rooty = y
            prex = 0
            prey = 0

        if i == 1 or i == 5 or i == 9 or i == 13 or i == 17:
            prex = rootx
            prey = rooty

        #
        if x == 0 and y == 0:
            prex = 0
            prey = 0
            continue

        # add new “if prex != 0 and prey != 0:” 是为了预防手腕关键点没有识别到？
        if prex != 0 and prey != 0:
            if (i > 0) and (i <= 4):
                cv2.line(im, (prex, prey), (x, y), line_colour[0], line_thickness, lineType=cv2.LINE_AA)
                cv2.circle(im, (x, y), circle_size, (0, 0, 255), -1)
                finger_num = 0
            if (i > 4) and (i <= 8):
                cv2.line(im, (prex, prey), (x, y), line_colour[1], line_thickness, lineType=cv2.LINE_AA)
                cv2.circle(im, (x, y), circle_size, (0, 255, 255), -1)
                finger_num = 1
            if (i > 8) and (i <= 12):
                cv2.line(im, (prex, prey), (x, y), line_colour[2], line_thickness, lineType=cv2.LINE_AA)
                cv2.circle(im, (x, y), circle_size, (0, 255, 0), -1)
                finger_num = 2
            if (i > 12) and (i <= 16):
                cv2.line(im, (prex, prey), (x, y), line_colour[3], line_thickness, lineType=cv2.LINE_AA)
                cv2.circle(im, (x, y), circle_size, (255, 255, 0), -1)
                finger_num = 3
            if (i > 16) and (i <= 20):
                cv2.line(im, (prex, prey), (x, y), line_colour[4], line_thickness, lineType=cv2.LINE_AA)
                cv2.circle(im, (x, y), circle_size, (255, 0, 0), -1)
                finger_num = 4
        else:
            if (i > 0) and (i <= 4):
                cv2.circle(im, (x, y), circle_size, line_colour[0], -1)
                finger_num = 0
            if (i > 4) and (i <= 8):
                cv2.circle(im, (x, y), circle_size, line_colour[1], -1)
                finger_num = 1
            if (i > 8) and (i <= 12):
                cv2.circle(im, (x, y), circle_size, line_colour[2], -1)
                finger_num = 2
            if (i > 12) and (i <= 16):
                cv2.circle(im, (x, y), circle_size, line_colour[3], -1)
                finger_num = 3
            if (i > 16) and (i <= 20):
                cv2.circle(im, (x, y), circle_size, line_colour[4], -1)
                finger_num = 4

        # cv2.putText(im, text=str(i), org=(x, y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.3,
        #             color=line_colour[finger_num], thickness=1)

        prex = x
        prey = y

    return im

def draw_text(points, im_ori):
    '''
    打印出gt中landmarks中的标记数字

    Parameters
    ----------
    (可参照上面的draw_2d_points函数输入)
    points
    im_ori

    Returns
    -------
    im: a image with test about landmarks
    '''
    im = im_ori.copy()
    finger_num = 0
    NUM_JOINTS = points.shape[0]
    for i in range(NUM_JOINTS):
        point = points[i]
        x = int(point[0])
        y = int(point[1])

        #
        if x == 0 and y == 0:
            continue

        # add new “if prex != 0 and prey != 0:” 是为了预防手腕关键点没有识别到？
        if (i > 0) and (i <= 4):
            finger_num = 0
        if (i > 4) and (i <= 8):
            finger_num = 1
        if (i > 8) and (i <= 12):
            finger_num = 2
        if (i > 12) and (i <= 16):
            finger_num = 3
        if (i > 16) and (i <= 20):
            finger_num = 4

        cv2.putText(im, text=str(i), org=(x, y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.3,
                    color=line_colour[finger_num], thickness=1)

    return im



# 返回box坐标
def coords_to_box(coords, box_factor=1.0):
    '''
    计算box边界并返回

    Parameters
    ----------
    coords
    box_factor

    Returns
    -------
    box: 返回box的坐标
    '''
    coord_min = np.min(coords, axis=0)
    coord_max = np.max(coords, axis=0)
    box_c = (coord_max + coord_min) / 2
    box_size = np.max(coord_max - coord_min) * box_factor

    x_left = int(box_c[0] - box_size / 2)
    y_top = int(box_c[1] - box_size / 2)
    x_right = int(box_c[0] + box_size / 2)
    y_bottom = int(box_c[1] + box_size / 2)

    box = [(x_left, y_top), (x_right, y_bottom)]
    return box

def L2Distance_calculation_YT3D(gt_kp_points, our_kp_points, image):
    '''
    计算21个关键点l2距离并返回

    Parameters
    ----------
    gt_kp_points
    our_kp_points
    image

    Returns
    -------

    '''
    dx = (gt_kp_points - our_kp_points)[:, 0]
    dy = (gt_kp_points - our_kp_points)[:, 1]
    sp = image.shape
    dx = dx / sp[1]
    dy = dy / sp[0]
    res = np.sum(np.hypot(dx, dy))
    return res

# 对比上面，这里会只读取gt中不为0的数据，对每个
def L2Distance_calculation_HFB(gt_kp_points, our_kp_points, image):
    '''
    计算不为0的所有关键点l2距离并返回

    Parameters
    ----------
    gt_kp_points
    our_kp_points
    image

    Returns
    -------

    '''
    NUM_Point = gt_kp_points.shape[0]
    gt_exist_kppoints = []
    our_exist_kppoint = []
    for i in range(NUM_Point):
        point = gt_kp_points[i]
        if point[0] != 0 and point[1] != 0:
            gt_exist_kppoints.append(point)
            our_exist_kppoint.append(our_kp_points[i])

    gt_kp_points = np.array(gt_exist_kppoints)
    our_kp_points = np.array(our_exist_kppoint)

    dx = (gt_kp_points - our_kp_points)[:, 0]
    dy = (gt_kp_points - our_kp_points)[:, 1]
    sp = image.shape
    dx = dx / sp[1]
    dy = dy / sp[0]

    res = np.sum(np.hypot(dx, dy)) / dx.shape[0]
    # res = np.sum(np.hypot(dx, dy))

    return res


def bb_iou(box_a, box_b):
    box_a = np.array(box_a).flatten()
    box_b = np.array(box_b).flatten()

    # determine the (x, y)-coordinates of the intersection rectangle
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])

    # compute the area of intersection rectangle
    inter_area = abs(max((x_b - x_a, 0)) * max((y_b - y_a), 0))
    if inter_area == 0:
        return 0

    # compute the area of both the prediction and ground-truth rectangles
    box_a_area = abs((box_a[2] - box_a[0]) * (box_a[3] - box_a[1]))
    box_b_area = abs((box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))

    # compute the intersection over union by taking the intersection area and dividing it by the sum of
    # prediction + ground-truth areas - the interesection area
    iou = inter_area / float(box_a_area + box_b_area - inter_area)

    # return the intersection over union value
    return iou


class VideoWriter(object):
    def __init__(self, save_name, cap, size=None):
        os.makedirs(os.path.dirname(save_name), exist_ok=True)
        # parameters of the video header
        fps = cap.get(cv2.CAP_PROP_FPS)
        # fps = 20
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







