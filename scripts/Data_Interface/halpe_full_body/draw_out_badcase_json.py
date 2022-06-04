import os

badcase_txt = r'E:\Data\landmarks\HFB\test\badcase.txt'
json_path = r'E:\Data\landmarks\HFB\HFB\annotations\person_keypoints_val2017.json'
save_path = r'E:\Data\landmarks\HFB\test\crop_badcase.json'


def main():
    # 得到所有badcase的图片名
    image_name_list = list()
    with open(badcase_txt, 'r') as f_txt:
        badcase_info = f_txt.readlines()
    for i in range(len(badcase_info)):
        image_name = os.path.split(badcase_info[i])[1].split('\n')[0]
        image_name_list.append(image_name)


if __name__ == '__main__':
    main()