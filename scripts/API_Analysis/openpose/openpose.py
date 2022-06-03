# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform
import argparse
import numpy as np
import time

line_colour = [(0, 0, 255), (0, 255, 255), (0, 255, 0), (255, 255, 0), (255, 0, 0)]
# 输出2d特征点
def draw_2d_points(points, im_ori, finger_num = 21):
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
    line_num = 5
    circle_num = 9

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
                cv2.line(im, (prex, prey), (x, y), line_colour[0], line_num, lineType=cv2.LINE_AA)
                cv2.circle(im, (x, y), circle_num, (0, 0, 255), -1)
                finger_num = 0
            if (i > 4) and (i <= 8):
                cv2.line(im, (prex, prey), (x, y), line_colour[1], line_num, lineType=cv2.LINE_AA)
                cv2.circle(im, (x, y), circle_num, (0, 255, 255), -1)
                finger_num = 1
            if (i > 8) and (i <= 12):
                cv2.line(im, (prex, prey), (x, y), line_colour[2], line_num, lineType=cv2.LINE_AA)
                cv2.circle(im, (x, y), circle_num, (0, 255, 0), -1)
                finger_num = 2
            if (i > 12) and (i <= 16):
                cv2.line(im, (prex, prey), (x, y), line_colour[3], line_num, lineType=cv2.LINE_AA)
                cv2.circle(im, (x, y), circle_num, (255, 255, 0), -1)
                finger_num = 3
            if (i > 16) and (i <= 20):
                cv2.line(im, (prex, prey), (x, y), line_colour[4], line_num, lineType=cv2.LINE_AA)
                cv2.circle(im, (x, y), circle_num, (255, 0, 0), -1)
                finger_num = 4
        else:
            if (i > 0) and (i <= 4):
                cv2.circle(im, (x, y), circle_num, line_colour[0], -1)
                finger_num = 0
            if (i > 4) and (i <= 8):
                cv2.circle(im, (x, y), circle_num, line_colour[1], -1)
                finger_num = 1
            if (i > 8) and (i <= 12):
                cv2.circle(im, (x, y), circle_num, line_colour[2], -1)
                finger_num = 2
            if (i > 12) and (i <= 16):
                cv2.circle(im, (x, y), circle_num, line_colour[3], -1)
                finger_num = 3
            if (i > 16) and (i <= 20):
                cv2.circle(im, (x, y), circle_num, line_colour[4], -1)
                finger_num = 4

        # cv2.putText(im, text=str(i), org=(x, y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.3,
        #             color=line_colour[finger_num], thickness=1)

        prex = x
        prey = y

    return im

def coord_to_box(coords, box_factor=3):
    coord_min = np.min(coords, axis=0)
    coord_max = np.max(coords, axis=0)
    box_c = (coord_max + coord_min) / 2.0
    box_size = np.max(coord_max - coord_min) * box_factor

    x_left = int(box_c[0] - box_size / 2.0)
    y_top = int(box_c[1] - box_size / 2.0)
    w = box_size
    h = w

    box = [x_left, y_top, w, h]
    return box

try:
    # Import Openpose (Windows/Ubuntu/OSX)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    try:
        # Windows Import
        if platform == "win32":
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append(dir_path + '/../../python/openpose/Release');
            os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/../../x64/Release;' +  dir_path + '/../../bin;'
            import pyopenpose as op
        else:
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append('../../python');
            # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
            # sys.path.append('/usr/local/python')
            from openpose import pyopenpose as op
    except ImportError as e:
        print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
        raise e

    # Flags
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", default="../../../examples/media/COCO_val2014_000000000241.jpg", help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
    parser.add_argument("--data_path", default="/workspace/cpfs-data/Data/test/new_data/aiyu/images/left", help=" ")
    parser.add_argument("--save_path", default="/workspace/nas-data/new_data/output/aiyu/left", help=" ")
    args = parser.parse_known_args()

    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = dict()
    params["model_folder"] = "../../../models/"
    params["hand"] = True
    params["hand_detector"] = 2
    params["body"] = 0

    save_path = args[0].save_path
    is_exists = os.path.exists(save_path)
    if not is_exists:
        os.makedirs(save_path)
        print('path of %s is build' % save_path)

    # Add others in path?
    for i in range(0, len(args[1])):
        curr_item = args[1][i]
        if i != len(args[1])-1: next_item = args[1][i+1]
        else: next_item = "1"
        if "--" in curr_item and "--" in next_item:
            key = curr_item.replace('-','')
            if key not in params:  params[key] = "1"
        elif "--" in curr_item and "--" not in next_item:
            key = curr_item.replace('-','')
            if key not in params: params[key] = next_item

    # Construct it from system arguments
    # op.init_argv(args[1])
    # oppython = op.OpenposePython()

    is_exists = os.path.exists(args[0].save_path)
    if not is_exists:
        os.mkdir(args[0].save_path)

    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    # Read image and face rectangle locations
    imageToProcess = cv2.imread(args[0].image_path)
    handRectangles = [
        # Left/Right hands person 0
        [
            op.Rectangle(50., 240., 400., 400.),
            op.Rectangle(0., 0., 0., 0.),
        ]
    ]

    id_list = []
    img_names = os.listdir(args[0].data_path)
    # for img_name in img_names:
    #     id = int(os.path.splitext(img_name)[0])
    #     id_list.append(id)
    img_names.sort()
    total_time = 0
    count = 0
    while 1:
        for index in range(len(img_names)):
            img_name = img_names[index]
            img_dir = os.path.join(args[0].data_path, img_name)
            start = time.time()
            imageToProcess = cv2.imread(img_dir)

            # Create new datum
            datum = op.Datum()
            datum.cvInputData = imageToProcess
            datum.handRectangles = handRectangles
            # Process and display image
            opWrapper.emplaceAndPop(op.VectorDatum([datum]))

            save_dir = os.path.join(args[0].save_path, img_name)

            # coords = datum.handKeypoints[1][0]   # 右手坐标
            coords = datum.handKeypoints[0][0]      # 左手坐标

            if np.mean(coords[:, 2]) < 0.05:
                # print(img_name + ' is loss')
                # op.Rectangle(135., 240., 400., 400.),
                # op.Rectangle(50., 240., 400., 400.),
                handRectangles = [
                    # Left/Right hands person 0
                    [
                        op.Rectangle(50., 240., 400., 400.),
                        op.Rectangle(0., 0., 0., 0.),
                    ]
                ]
                cv2.imwrite(save_dir, imageToProcess)
                continue
            # print(img_name)
            box = coord_to_box(coords[:, 0:2])
            handRectangles = [
                # Left/Right hands person 0
                [
                    op.Rectangle(box[0], box[1], box[2], box[3]),
                    op.Rectangle(0., 0., 0., 0.),
                ]
            ]
            end = time.time()
            running_time = (end - start) * 1000
            total_time = total_time + running_time
            count += 1
            print(count)
            print(f'This python process id is: {os.getpid()}')
            if count == 2000:
                break
            coords_img = draw_2d_points(coords, imageToProcess)

            # cv2.imwrite(save_dir, coords_img)
            cv2.waitKey(0)
        if count == 2000:
            break
    avg_time = total_time / 2000
    FPS = 1000 / avg_time
    print(f'Avage time is:{avg_time}')
    print(f'FPS is:{FPS}')

except Exception as e:
    print(e)
    sys.exit(-1)