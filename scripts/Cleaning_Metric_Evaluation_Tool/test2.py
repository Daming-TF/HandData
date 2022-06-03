import os
data_path = r'E:\L2_eval\new_data\images\test2017'
record_txt_path = r'E:\L2_eval\new_data\badcase\1.txt'

filenames = os.listdir(data_path)
with open(record_txt_path, 'a') as f:
    for filenmae in filenames:
        img_path = os.path.join(data_path, filenmae)
        f.write(img_path)
        f.write('\n')
