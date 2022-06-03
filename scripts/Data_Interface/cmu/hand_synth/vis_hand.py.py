import json
import numpy as np
import cv2
import os
from tools import draw_2d_points


def main(path):
    filenames = os.listdir(path)
    for filename in filenames:
        if filename.endswith('.jpg'):
            json_path = os.path.join(path, (filename.split(".")[0]+".json"))
            image_path = os.path.join(path, filename)
            image = cv2.imread(image_path)
            with open(json_path, "r") as f:
                data = json.load(f)
                hand_pts = data['hand_pts']
                hand_pts = np.array(hand_pts)
                # print(hand_pts)
                im = draw_2d_points(hand_pts, image, 21)
                cv2.imshow("check gt", im)
                print(filename)

            if cv2.waitKey(0) == 27:
                    exec("Esc clicked!")


if __name__ == "__main__":
    path = \
        r"F:\image\CMU\hand_labels_synth\hand_labels_synth\our_train"
    main(path)