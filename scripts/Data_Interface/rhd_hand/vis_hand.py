""" Basic example showing samples from the dataset.

    Chose the set you want to see and run the script:
    > python view_samples.py

    Close the figure to see the next sample.
"""
from __future__ import print_function, unicode_literals

import pickle
import os
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
from tools import draw_2d_points, draw_text

# chose between training and evaluation set
# set = 'training'
# set = 'evaluation'
set = r'F:\image\Rendered Handpose Dataset Dataset\RHD_v1-1\RHD_published_v2\training'
path = r"F:\image\Rendered Handpose Dataset Dataset\RHD_v1-1\RHD_published_v2\training\anno_training.pickle"


def coordinate_normalization(points):
    kp = np.zeros(63).reshape(21,3)
    kp[0] = points[0]
    kp[1] = points[4]
    kp[2] = points[3]
    kp[3] = points[2]
    kp[4] = points[1]
    kp[5] = points[8]
    kp[6] = points[7]
    kp[7] = points[6]
    kp[8] = points[5]
    kp[9] = points[12]
    kp[10] = points[11]
    kp[11] = points[10]
    kp[12] = points[9]
    kp[13] = points[16]
    kp[14] = points[15]
    kp[15] = points[14]
    kp[16] = points[13]
    kp[17] = points[20]
    kp[18] = points[19]
    kp[19] = points[18]
    kp[20] = points[17]
    return kp



# auxiliary function
def depth_two_uint8_to_float(top_bits, bottom_bits):
    """ Converts a RGB-coded depth into float valued depth. """
    depth_map = (top_bits * 2**8 + bottom_bits).astype('float32')
    depth_map /= float(2**16 - 1)
    depth_map *= 5.0
    return depth_map



def main():
    # load annotations of this set
    # with open(os.path.join(set, 'anno_%s.pickle' % set), 'rb') as fi:
    #     anno_all = pickle.load(fi)
    with open(path, 'rb') as fi:
        anno_all = pickle.load(fi)

    # iterate samples of the set
    for sample_id, anno in anno_all.items():
        print(sample_id)
        # load data
        image = cv2.imread(os.path.join(set, 'color', '%.5d.png' % sample_id))
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(set, 'mask', '%.5d.png' % sample_id))
        depth = cv2.imread(os.path.join(set, 'depth', '%.5d.png' % sample_id))

        # process rgb coded depth into float: top bits are stored in red, bottom in green channel
        depth = depth_two_uint8_to_float(depth[:, :, 0], depth[:, :, 1])  # depth in meters from the camera

        a = anno['uv_vis']
        print(a)
        # get info from annotation dictionary
        kp_coord_uv = anno['uv_vis'][:, :2]  # u, v coordinates of 42 hand keypoints, pixel
        kp_visible = (anno['uv_vis'][:, 2] == 1)  # visibility of the keypoints, boolean
        kp_coord_xyz = anno['xyz']  # x, y, z coordinates of the keypoints, in meters
        camera_intrinsic_matrix = anno['K']  # matrix containing intrinsic parameters

        # Project world coordinates into the camera frame
        kp_coord_uv_proj = np.matmul(kp_coord_xyz, np.transpose(camera_intrinsic_matrix))
        kp_coord_uv_proj = kp_coord_uv_proj[:, :2] / kp_coord_uv_proj[:, 2:]

        kp = coordinate_normalization(anno['uv_vis'][:21])
        im = draw_2d_points(kp[:21], image)
        im_txt = draw_text(kp[:21], image)
        im = cv2.resize(im, (600, 600), interpolation=cv2.INTER_LINEAR)
        image = cv2.resize(image, (600, 600), interpolation=cv2.INTER_LINEAR)
        im_txt = cv2.resize(im_txt, (600, 600), interpolation=cv2.INTER_LINEAR)
        canvas1 = np.hstack([image, im, im_txt])
        cv2.imshow("check RHD", canvas1)
        # Visualize data
        # fig = plt.figure(1)
        # ax1 = fig.add_subplot('111')
        # ax2 = fig.add_subplot('222')
        # ax3 = fig.add_subplot('223')
        # ax4 = fig.add_subplot('224', projection='3d')

        # ax1.imshow(image)
        # ax1.plot(kp_coord_uv[kp_visible, 0], kp_coord_uv[kp_visible, 1], 'ro')
        #ax1.plot(kp_coord_uv_proj[kp_visible, 0], kp_coord_uv_proj[kp_visible, 1], 'gx')
        # ax2.imshow(depth)
        # ax3.imshow(mask)
        # ax4.scatter(kp_coord_xyz[kp_visible, 0], kp_coord_xyz[kp_visible, 1], kp_coord_xyz[kp_visible, 2])
        # ax4.view_init(azim=-90.0, elev=-90.0)  # aligns the 3d coord with the camera view
        # ax4.set_xlabel('x')
        # ax4.set_ylabel('y')
        # ax4.set_zlabel('z')

        # plt.show()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            exit(0)

if __name__ == "__main__":
    main()