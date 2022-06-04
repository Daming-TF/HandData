# coding=UTF-8
import cv2
import glob, os
import numpy as np
import re
import fnmatch
import pickle
import random

from library.tools import draw_2d_points


def natural_sort(l):
    # l表示所有图片文件路径的list对象
    # isdigit()方法检测字符串是否只由数字组成
	# lowwer()方法转换字符串中所有大写字符为小写
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    # re.split(pattern, string, maxsplit=0, flags=0)
    # 	pattern：相当于str.split()中的sep，分隔符的意思，不但可以是字符串，也可以为正则表达式: '[ab]'
    # 表示的意思就是取a和b的任意一个值（可参考： https://docs.python.org/3/library/re.html?highlight=re%20split#re.split ）
    # 	string：要进行分割的字符串
    # 	maxsplit：分割的最大次数，这个参数和str.split()中有点不一样：
	# "+"表示匹配前面的子表达式一次或多次。例如，'zo+' 能匹配 "zo" 以及 "zoo"，但不能匹配 "z"。+ 等价于 {1,}。
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    # sorted() 函数对所有可迭代的对象进行排序操作。
	# l 表示可迭代对象
	# kep 主要是用来进行比较的元素，只有一个参数，具体的函数的参数就是取自于可迭代对象中，指定可迭代对象中的一个元素来进行排序
    return sorted(l, key = alphanum_key)


def recursive_glob(rootdir='.', pattern='*'):
    matches = []
    # os.walk遍历目标文件rootdir，但会三个参数
    # root 指向当前正在遍历的文件夹本身地址
    # dirs 是一个list， 内容为该文件夹中所有的目录文件夹
    # files 同样为一个list， 内容是该文件夹中所有的文件
    # 对 filenames 列表进行过滤，返回 filenames 列表中匹配 pattern 的文件名组成的子集合
    '''
    fnmatch 模块匹配文件名的模式使用的就是 UNIX shell 风格，其支持使用如下几个通配符：
        *：可匹配任意个任意字符。
        ？：可匹配一个任意字符。
        [字符序列]：可匹配中括号里字符序列中的任意字符。该字符序列也支持中画线表示法。比如 [a-c] 可代表 a、b 和 c 字符中任意一个。
        [!字符序列]：可匹配不在中括号里字符序列中的任意字符。
    '''
    for root, dirnames, filenames in os.walk(rootdir):
        for filename in fnmatch.filter(filenames, pattern):
            # 获取所有匹配"*_webcam_[0-9]*"的文件
            matches.append(os.path.join(root, filename))

    return matches


def main():
    pathToDataset = r"F:/image/Multiview Hand Pose/archive/annotated_frames"
    for i in range(1, 2):
        # read the color frames
        path = pathToDataset + "/data_" + str(i) + "/"
        # 返回一个list，内容为获取的所有匹配"*_webcam_[0-9]*"的文件路径
        colorFrames = recursive_glob(path, "*_webcam_[0-9]*")
        colorFrames = natural_sort(colorFrames)
        for j in range(len(colorFrames)):
            print(colorFrames[j])
            toks1 = colorFrames[j].split("/")
            toks2 = toks1[-1].split("_")
            txtPath = r"F:/image/Multiview Hand Pose/archive/projections_2d/data_"+str(i)+"/"+toks2[0]+"_jointsCam_"+toks2[2].split(".")[0]+".txt"

            result = []
            with open(txtPath, 'r') as f:
                for line in f:
                    result.append(list(line.strip('\n').split(' ')))
            kp = np.array(result)[:,1:3]
            kp = kp.astype(float)
            img = cv2.imread(colorFrames[j])
            im = draw_2d_points(kp, img, 21)
            cv2.imshow("check pts", im)
            if cv2.waitKey(0) == 27:
                exec("Esc clicked!")


if __name__ == "__main__":
    main()
