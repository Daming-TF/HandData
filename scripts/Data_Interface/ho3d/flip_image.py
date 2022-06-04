import os
import cv2
from tqdm import tqdm

data_dir = r'G:\imgdate2\HO3D_v3\HO3D_from_whole_body\images\val2017'

file_names = os.listdir(data_dir)
for i in tqdm(range(len(file_names))):
    file_name = file_names[i]
    if file_name.endswith('.jpg'):
        id = os.path.splitext(file_name)[0]
        if int(id) <= 1300450:      # 1300450 val       # 1337249 train
            image_dir = os.path.join(data_dir, file_name)
            img = cv2.imread(image_dir)
            img = cv2.flip(img, 1)
            cv2.imwrite(image_dir, img)
