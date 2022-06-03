"""
    统计每个数据集手目标最小框（×1.5）的分辨率分布
"""
import json
from tqdm import tqdm

json_dir = r'E:\Data\landmarks\FreiHAND_pub_v2\badcase\FH.json'

count_64, count_96, count_128, count_160, count_192, count_224, count_256, count_more_than256 = 0, 0, 0, 0, 0, 0, 0, 0
with open(json_dir, 'r')as f:
    json_data = json.load(f)
    json_annotations = json_data['annotations']
    for index in tqdm(range(len(json_annotations))):
        anno_info = json_annotations[index]
        bbox = anno_info['bbox']
        box_w = bbox[2]
        box_h = bbox[3]
        assert (box_h == box_h)
        if box_w <= 64:
            count_64 += 1
        elif box_w <= 96:
            count_96 += 1
        elif box_w <= 128:
            count_128 += 1
        elif box_w <= 160:
            count_160 += 1
        elif box_w <= 192:
            count_192 += 1
        elif box_w <= 224:
            count_224 += 1
        elif box_w <= 256:
            count_256 += 1
        elif box_w > 256:
            count_more_than256 += 1

    print(f'Finish loaded all the date NUM:{len(json_annotations)}')
    print(f'''
    the size ∈(0,64] :{count_64}
    the size ∈(64,96] :{count_96}
    the size ∈(96,128] :{count_128}
    the size ∈(128,160] :{count_160}
    the size ∈(160,192] :{count_192}
    the size ∈(192,224] :{count_224}
    the size ∈(224,256] :{count_256}
    the size ∈(256,∞) :{count_more_than256}
    
    ''')


