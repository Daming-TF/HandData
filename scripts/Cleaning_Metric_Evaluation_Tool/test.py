import os
import random
import cv2

badcase_path = r'E:\L2_eval\new_data\badcase\badcase.txt'
data_path = r'G:\test_data\new_data\crop_images'
save_path = r'E:\L2_eval\new_data\images'
record_txt_path = r'E:\L2_eval\new_data\images\1.txt'
badcase_list = list()

with open(badcase_path, 'r') as f:
    line = f.readline()
    while line:
        badcase = os.path.split(line)[1].replace('\n', '')
        badcase_list.append(badcase)
        line = f.readline()

filenames = os.listdir(data_path)
count = 0
for filename in filenames:
    if (filename in badcase_list):
        print(filename)
        continue

    if random.randint(0, 4):
        continue
    img_path = os.path.join(data_path, filename)
    image = cv2.imread(img_path)
    save_dir = os.path.join(save_path, filename)
    cv2.imwrite(save_dir, image)
    with open(record_txt_path, 'a') as f:
        f.write(img_path + "\n")
    count += 1
    if count == 1000:
        break
