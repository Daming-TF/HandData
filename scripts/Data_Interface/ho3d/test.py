import copy
import os
import pickle
import numpy as np
import cv2
from tools import draw_2d_points

def canonical_coordinates(points):
    kp = copy.deepcopy(points)
    kp[0] = points[0]
    kp[1] = points[13]
    kp[2] = points[14]
    kp[3] = points[15]
    kp[4] = points[16]
    kp[5] = points[1]
    kp[6] = points[2]
    kp[7] = points[3]
    kp[8] = points[17]
    kp[9] = points[4]
    kp[10] = points[5]
    kp[11] = points[6]
    kp[12] = points[18]
    kp[13] = points[10]
    kp[14] = points[11]
    kp[15] = points[12]
    kp[16] = points[19]
    kp[17] = points[7]
    kp[18] = points[8]
    kp[19] = points[9]
    kp[20] = points[20]
    return kp


def load_pickle_data(f_name):
    """ Loads the pickle data """
    if not os.path.exists(f_name):
        raise Exception('Unable to find annotations picle file at %s. Aborting.'%(f_name))
    with open(f_name, 'rb') as f:
        try:
            pickle_data = pickle.load(f, encoding='latin1')
        except:
            pickle_data = pickle.load(f)

    return pickle_data

def projectPoints(xyz, K):
    """ Project 3D coordinates into image space. """
    xyz = np.array(xyz)
    K = np.array(K)
    uv = np.matmul(K, xyz.T).T
    return uv[:] / uv[:, -1:]

def get_intrinsics(filename):
    with open(filename, 'r') as f:
        line = f.readline()
    line = line.strip()
    items = line.split(',')
    for item in items:
        if 'fx' in item:
            fx = float(item.split(':')[1].strip())
        elif 'fy' in item:
            fy = float(item.split(':')[1].strip())
        elif 'ppx' in item:
            ppx = float(item.split(':')[1].strip())
        elif 'ppy' in item:
            ppy = float(item.split(':')[1].strip())

    camMat = np.array([[fx, 0, ppx], [0, fy, ppy], [0, 0, 1]])
    return camMat

# K = get_intrinsics(os.path.join(
#         r"G:\Model_output\image\test\calibration\AP1\calibration", 'cam_1_intrinsics.txt')).tolist()
K = get_intrinsics(os.path.join(
        r"F:\image\HO3D\HO3D_v3\HO3D_v3\HO3D_v3\calibration\ShSu1\calibration", 'cam_4_intrinsics.txt')).tolist()

# optPickData = load_pickle_data(r"G:\Model_output\image\test\evaluation\AP11\meta\0007.pkl")
optPickData = load_pickle_data(r"F:\image\HO3D\HO3D_v3\HO3D_v3\HO3D_v3\train\ShSu14\meta\1729.pkl")

print(optPickData.keys())
rightHandJointLocs = optPickData['handJoints3D']
# if np.all(rightHandJointLocs == None):
if np.all(rightHandJointLocs == None):
    exit(0)
print(rightHandJointLocs)

kp = projectPoints(rightHandJointLocs, K)
print(kp)
print("\n")
kp = canonical_coordinates(kp)
print(kp)

# img = cv2.imread(r"G:\Model_output\image\test\evaluation\AP11\rgb\0007.jpg")
img = cv2.imread(r"G:\image\HO3D_v3\train\ShSu14\rgb\1729.jpg")

img = cv2.flip(img, 1)
print(img.shape)
im = draw_2d_points(kp, img, 21)
im = cv2.resize(im, (700, 700), interpolation=cv2.INTER_LINEAR)

cv2.imshow("test", im)
if cv2.waitKey(0) & 0xFF == ord('q'):
    exit(0)

