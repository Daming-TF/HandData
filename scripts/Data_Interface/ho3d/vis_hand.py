import os
import pickle
import numpy as np
import cv2

from library.tools import draw_2d_points


def canonical_coordinates(points):
    kp = np.ones(63).reshape(21,3)

    kp[0,] = points[0]
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


def main():
    data_path = r"F:\image\HO3D\HO3D_v3\HO3D_v3\HO3D_v3\evaluation\SB13\rgb"
    pickle_dir = r"F:\image\HO3D\HO3D_v3\HO3D_v3\HO3D_v3\evaluation\SB13\meta"
    calibration_dir = r"F:\image\HO3D\HO3D_v3\HO3D_v3\HO3D_v3\calibration\SB1\calibration\cam_3_intrinsics.txt"
    filenames = os.listdir(data_path)
    for i in range(len(filenames)):
        id, _ = os.path.splitext(filenames[i])
        img_path = os.path.join(data_path, filenames[i])
        print(img_path)
        pickle_path = os.path.join(pickle_dir, str(id).zfill(4)+".pkl")
        print(pickle_path)
        K = get_intrinsics(calibration_dir).tolist()

        optPickData = load_pickle_data(pickle_path)
        print(optPickData.keys())
        rightHandJointLocs = optPickData['handJoints3D']
        if np.all(rightHandJointLocs == 0) or np.all(rightHandJointLocs == None):
            continue
        if not rightHandJointLocs.shape == (21, 3):
            continue
        print(id)
        kp = projectPoints(rightHandJointLocs, K)
        kp = canonical_coordinates(kp)

        img = cv2.imread(img_path)
        img = cv2.flip(img, 1)
        print(img.shape)

        im = draw_2d_points(kp, img, 21)
        im = cv2.resize(im, (700, 700), interpolation=cv2.INTER_LINEAR)

        cv2.imshow("test", im)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            exit(0)


if __name__ == "__main__":
    main()
