import copy
import os.path
import numpy as np
import matplotlib.pyplot as plt

def plot_PR(x, index_data_list, badcase_list, i):
    print(f"start to plot the index_{i}")
    # 正样本为badcase
    precision_max = 0
    record_th = 0
    precision_list = list()
    recall_list = list()
    for th in x:
        TP, TN, FP, FN = 0.0, 0.0, 0.0, 0.0
        for data in index_data_list:
            filename = data['filename']
            score = data['score']
            flag = 'negatives'
            for name in badcase_list:
                if filename == name:
                    flag = 'positives'
            if score > th and flag == 'positives':
                TP += 1
            elif score < th and flag == 'negatives':
                TN += 1
            elif score > th and flag == 'negatives':
                FP += 1
            elif score < th and flag == 'positives':
                FN += 1

        precision = TP / (TP + FP)
        precision_list.append(precision)
        recall = TP / (TP + FN)
        recall_list.append(recall)
        if recall >= 0.95 and precision > precision_max:
            precision_max = precision
            record_th = th

    with open(txt_dir, 'a') as f:
        f.writelines(str(precision_list[9:]))
        f.write('\n')
        f.writelines(str(recall_list[9:]))
        f.write('\n')
    print(precision_list)
    print(recall_list)
    print(f'The max precision(fixed the recall > 0.95) is : >>{precision_max}<<')
    print(f'The threshold is : >>{record_th}<<')

    plt.title('Metric-EVAL')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    # 调整x,y坐标范围
    plt.xlim(0.2, 1)
    plt.ylim(0.92, 1.01)
    # 显示背景的网格线
    plt.grid(True)
    plt.plot(np.array(recall_list), np.array(precision_list),
             linestyle=linestyle[i-1], color=color[i-1], label=f'index{i}')

def check_distribution(scores):
    num_0, num_1, num_2, num_3, num_4, num_5, num_6, num_7, num_8, num_9, num_10, num_15, num_20 = \
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    for score in scores:
        if score < 2:
            num_0 += 1
        elif score < 4 and score >2:
            num_1 += 1
        elif score < 6 and score >4:
            num_2 += 1
        elif score < 8 and score >6:
            num_3 += 1
        elif score < 10 and score > 8:
            num_4 += 1
        elif score < 12 and score > 10:
            num_5 += 1
        elif score < 14 and score > 12:
            num_6 += 1
        elif score < 16 and score > 14:
            num_7 += 1
        elif score < 18 and score > 16:
            num_8 += 1
        elif score < 20 and score > 18:
            num_9 += 1
        # elif score < 15 and score > 10:
        #     num_10 += 1
        # elif score < 20 and score > 15:
        #     num_15 += 1
        elif score > 20:
            num_20 += 1

    print(f'''
    (0-2): {num_0}, (2-4): {num_1}, (4-6): {num_2}
    (6-8): {num_3}, (8-10): {num_4}, (10-12): {num_5}
    (12-14): {num_6}, (14-16): {num_7}, (16-18): {num_8}
    (18-20): {num_9}, (>20): {num_20}
    ''')

def check_score_max(info_list):
    scores = []
    for info in info_list:
        score = info['score']
        scores.append(score)
    scores.sort(reverse=True)
    check_distribution(scores)
    return scores[0]/100.0

def open_txt(txt_path):
    print(txt_path)
    info_list = []
    block_dict = {}
    scores = list()
    with open(txt_path, 'r') as f:
        line = f.readline()
        while line:
            filename = line.split('\t')[0]
            score = float(line.split('\t')[1].replace('\n', ''))
            block_dict['filename'] = filename
            block_dict['score'] = score
            scores.append(score)
            info_list.append(copy.deepcopy(block_dict))
            line = f.readline()

    scores.sort(reverse = True)
    max = scores[0]
    for i in range(len(info_list)):
        old_score = info_list[i]['score']
        if old_score > max:
            print('error!')
        info_list[i]['score'] = float(old_score)/float(max)*100.0

    return info_list


def main():
    index_list = [{}]
    badcase_list = list()

    with open(badcase_path, 'r') as f:
        line = f.readline()
        while line:
            badcase = os.path.split(line)[1].replace('\n', '')
            badcase_list.append(badcase)
            line = f.readline()

    a = list(np.arange(0, 1, 0.02))
    b = list(np.arange(1, 10, 0.1))
    c = list(np.arange(10, 100, 1))

    classification_threshold = a+b+c

    for i in range(1,txt_num+1):
        exec("index{} = open_txt(txt_index{}_path)".format(i, i))
        index_list.append(locals()[f'index{i}'])

        score = check_score_max(index_list[i])
        if not score == 1:
            print('error:score is Size error')
            exit(1)
    # # plot_PR(x, index_data_list, badcase_list, i):
    # for i in range(len(index_list)-1):
    #     index_data = index_list[i+1]
    #     plot_PR(classification_threshold, index_data, badcase_list, i+1)
    # # plt.legend()函数的作用是给图像加图例
    # plt.legend(loc="best")
    # plt.savefig(r'D:\My Documents\Desktop\毕设资料\清洗评估算法\伪效果2.png')
    # plt.show()

if __name__ == '__main__':
    # new data
    txt_index1_path = r'E:\L2_eval\new_data\output\index1\new_data\images\test2017\record.txt'
    txt_index2_path = r'E:\L2_eval\new_data\output\index2\new_data\images\test2017\record.txt'
    txt_index3_path = r'E:\L2_eval\new_data\output\index3\new_data\images\test2017\record.txt'
    txt_index4_path = r'E:\L2_eval\new_data\output\index4\new_data\images\test2017\record.txt'
    txt_index5_path = r'E:\L2_eval\new_data\output\index5\new_data\images\test2017\record.txt'
    txt_index6_path = r'E:\L2_eval\new_data\output\index5-1-1\new_data\images\test2017\record.txt'
    txt_index7_path = r'E:\L2_eval\new_data\output\index5-1-2\new_data\images\test2017\record.txt'
    txt_index8_path = r'E:\L2_eval\new_data\output\index5-1-3\new_data\images\test2017\record.txt'
    txt_index9_path = r'E:\L2_eval\new_data\output\index5-2-1\new_data\images\test2017\record.txt'
    txt_index10_path = r'E:\L2_eval\new_data\output\index5-2-2\new_data\images\test2017\record.txt'
    # txt_index5_path = r'E:\L2_eval\new_data\output\index5-1-5\new_data\images\test2017\record.txt'
    badcase_path = r'E:\L2_eval\new_data\badcase\1.txt'

    # old data
    # txt_index1_path = r'E:\L2_eval\old_data\output\index1\L2_eval\images\test2017\record.txt'
    # txt_index2_path = r'E:\L2_eval\old_data\output\index2\L2_eval\images\test2017\record.txt'
    # txt_index3_path = r'E:\L2_eval\old_data\output\index3\L2_eval\images\test2017\record.txt'
    # txt_index4_path = r'E:\L2_eval\old_data\output\index4\L2_eval\images\test2017\record.txt'
    # txt_index5_path = r'E:\L2_eval\old_data\output\index5\L2_eval\images\test2017\record.txt'
    # badcase_path = r'E:\L2_eval\old_data\badcase\total_badcase.txt'

    txt_num = 10
    color = ['gray', 'lightsteelblue', 'palevioletred', 'salmon', 'cyan']
    linestyle = ['-', '-', '-.', '--', ':']
    txt_dir = r'D:\My Documents\Desktop\1.txt'
    main()