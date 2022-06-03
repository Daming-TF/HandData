import os
import cv2
import random
txt_path = r'E:\L2_eval\new_data\badcase\badcase.txt'
data_path = r'G:\test_data\new_data\crop_images'
save_path = r'E:\L2_eval\new_data\images'
total_badcase_txt_path = r''
with open(txt_path, "r") as f:  # 打开文件
    # data = f.read()  # 读取文件
    # print(data)
    # print(type(data))
    line = f.readline()
    count = 0
    while line:
        i = random.randint(0, 10)
        if i:
            continue
        filename = os.path.split(line)[1].replace('\n', '')
        print(filename)
        img_name = os.path.join(data_path, filename)
        img = cv2.imread(img_name)
        save_dir = os.path.join(save_path, filename)
        print(save_dir)
        cv2.imwrite(save_dir, img)
        with open(total_badcase_txt_path,'a') as total_f:
            total_f.writelines(line)
        count += 1
        if count == 500:
            break
        line = f.readline()
    print("succeed!")
