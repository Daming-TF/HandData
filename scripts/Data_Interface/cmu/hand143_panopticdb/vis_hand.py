import json
import numpy as np
import cv2
import os
from tools import draw_2d_points


def main(img_path, json_path):
    with open(json_path, "r") as f:
        json_infos = json.load(f)
        hands = json_infos['root']
        nums_hands = len(hands)
        for i in range(nums_hands):
            img_name = hands[i]['img_paths'].strip("")
            img_name = img_name.split('/')[1]
            kp = np.array(hands[i]['joint_self'])

            filename = os.path.join(img_path, img_name)
            print(filename)
            image = cv2.imread(filename)

            im = draw_2d_points(kp, image, 21 )
            # im = cv2.resize(im, (500, 500), interpolation=cv2.INTER_LINEAR)
            cv2.imshow("check gt", im)
            if cv2.waitKey(0) == 27:
                    exec("Esc clicked!")


if __name__ == "__main__":
    img_path = r"F:\image\CMU\hand143_panopticdb\hand143_panopticdb\imgs"
    json_path = r"F:\image\CMU\hand143_panopticdb\hand143_panopticdb\hands_v143_14817.json"

    main(img_path, json_path)