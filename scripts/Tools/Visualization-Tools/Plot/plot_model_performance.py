import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from adjustText import adjust_text


pro_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(pro_dir)


def convert_str_to_num(arr):
    # Convert K, M, G string to float number.
    data = list()
    num_arr = arr.copy()
    for i, f in enumerate(num_arr):
        data.append(float(f[0]))
    return num_arr


def get_name_and_color(resolutions, alphas, decoders, models=None):
    ihd_colors = np.array(
        [
            [229, 20, 0],  # red
            [96, 169, 23],  # green
            [27, 161, 226],  # cyan
            [240, 163, 10],  # amber
            [105, 0, 255],  # indigo
            [100, 118, 135],
        ],  # brown
        dtype=np.float32,
    )

    crd_colors = np.array(
        [
            [250, 104, 0],  # orange
            [0, 138, 0],  # emerald
            [0, 80, 239],  # cobalt
            [227, 200, 0],  # yellow
            [170, 0, 255],  # violet
            [118, 96, 138],
        ],  # taupe
        dtype=np.float32,
    )

    # Preprocessing
    exp_names = []
    colors = []
    for i, (resolution, alpha, decoder, model) in enumerate(
        zip(resolutions, alphas, decoders, models)
    ):
        if decoder[0] == "CoarseRefineDecoder":
            decoder = "CRD"
            candi_color = crd_colors
        elif decoder[0] == "Regressor":
            decoder = "IHD"
            candi_color = ihd_colors
        else:
            decoder = "SD"
            candi_color = 0.5 * (crd_colors + ihd_colors)

        transparent = 0.8
        resolution = resolution[0][: len(resolution[0]) // 2]

        # if resolution == "256":
        #     color = candi_color[0] / 255.0
        # elif resolution == "224":
        #     color = candi_color[1] / 255.0
        # elif resolution == "192":
        #     color = candi_color[2] / 255.0
        # elif resolution == "160":
        #     color = candi_color[3] / 255.0
        # elif resolution == "128":
        #     color = candi_color[4] / 255.0
        # else:  # 96
        #     color = candi_color[5] / 255.0
        if decoder == "CRD":
            color = np.array([106.0, 90.0, 205.0]) / 255.0
        elif decoder == "IHD":
            color = np.array([255.0, 106.0, 106.0]) / 255.0
        elif decoder == "SD":
            color = np.array([119.0, 136.0, 153.0]) / 255.0

        if "Full" in model[0]:
            model_name = "MP_Full"
        elif "Lite" in model[0]:
            model_name = "Lite"
        else:
            model_name = "MF"

        # alpha[0] = float(alpha[0])
        if model_name is None:
            name = "-".join(
                [
                    str(alpha[0]) if isinstance(alpha[0], float) else "MF",
                    '-'+resolution,
                    '-'+decoder,
                ]
            )
        else:
            name = "-".join(
                [
                    model_name,
                    resolution,
                    str(alpha[0]) if isinstance(alpha[0], float) else "",
                    decoder,
                ]
            )

        exp_names.append(name.replace("--", "-"))
        colors.append((*color, transparent))        # 颜色是根据输入分辨率决定的

    return exp_names, colors


def main():
    # Read data from excel,engine:指定excel处理引擎
    exp_data = pd.read_excel(
        r"D:\My Documents\Desktop\test.xlsx",
        engine="openpyxl",
    )

    # DataFrame创造数据框
    # 第一个参数是存放在DataFrame里的数据，第二个参数index就是之前说的行名，第三个参数columns是之前说的列名
    # 这里把excel数据中所有需要的数据转化为列表单元
    models = pd.DataFrame(exp_data, columns=["Model"]).values.tolist()
    resolutions = pd.DataFrame(exp_data, columns=["Resolution"]).values.tolist()
    alphas = pd.DataFrame(exp_data, columns=["Alpha"]).values.tolist()
    decoders = pd.DataFrame(exp_data, columns=["Decoder"]).values.tolist()
    params = pd.DataFrame(exp_data, columns=["#params"]).values.tolist()
    mflops = pd.DataFrame(exp_data, columns=["MFlops"]).values.tolist()
    f1 = pd.DataFrame(exp_data, columns=["F1"]).values.tolist()

    exp_names, colors = get_name_and_color(resolutions, alphas, decoders, models)
    mflops = convert_str_to_num(mflops)
    params = convert_str_to_num(params)
    f1 = convert_str_to_num(f1)
    # 这里计算min和max是为了后面定义画图区间
    min_f1, max_f1 = max(30, np.amin(f1) - 5), int(np.amax(f1) + 5)
    # min_mflops, max_mflops = (
    #     max(0, np.amin(mflops) - 100),
    #     max(np.amax(mflops) + 100, 10e2),
    # )
    min_mflops, max_mflops = (
        max(0, np.amin(mflops) - 30),
        max(np.amax(mflops) + 100, 10e2),
    )

    # Plot
    # plt.rcParams:使用rc配置文件来自定义图形的各种默认属性
    # 设置图像显示大小
    plt.rcParams["figure.figsize"] = [40, 20]
    # 字体大小
    plt.rcParams["font.size"] = 50      # 24

    fig, ax = plt.subplots()        # 等价于fig, ax = plt.subplots(11)
    # params = [p[0] / 1000 for p in params]
    params = [p[0]*3 / 1000 for p in params]
    # scatter画散点图
    # 第一个参数为x；第二个参数为y；marker：标记样式；s：表示的是大小；c：表示的是色彩或颜色序列
    ax.scatter(mflops, f1, marker="o", s=params, c=colors)
    ax.set_xscale("log")  # FLOPs are in log scale设置缩放比例

    texts = []
    for i, exp in enumerate(exp_names):
        texts.append(plt.text(mflops[i][0], f1[i][0], exp, fontsize=30))

    # # Adjust text to avoid overlapping
    # adjust_text(
    #     texts, expand_objects=(2, 2), arrowprops=dict(arrowstyle="->", color="r", lw=2)
    # )

    plt.xlabel("MFLOPs")
    plt.ylabel("F1-Score")
    ax.grid(True)

    plt.xlim((min_mflops, max_mflops))
    plt.ylim((min_f1, max_f1))  # f1 score
    plt.title("")

    # plt.show()
    plt.savefig(
        r"D:\My Documents\Desktop\12.png"
    )
    plt.close()


if __name__ == "__main__":
    main()
