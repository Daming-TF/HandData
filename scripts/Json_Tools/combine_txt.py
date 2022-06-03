import os

def main(path):
    testTxt_list = []
    trainTxt_list = []
    valTxt_list = []
    txt_list = []
    filenames = os.listdir(path)

    a = int
    for mode in ["train", "val", "test"]:
        for filename in filenames:
            file_path = os.path.join(path, filename)
            if not filename.find(mode) == -1:
                exec ("{}Txt_list.append('{}')".format(mode,filename))
            else:
                continue
        if mode == "train":
            txt_list = trainTxt_list
        elif mode == "val":
            txt_list = valTxt_list
        elif mode == "test":
            txt_list = testTxt_list
        if len(txt_list) == 0:
            continue
        file_num = len(txt_list)
        infoes = list()
        for index in range(file_num):
            txt_name = txt_list[index]
            print(txt_name)
            file_path = os.path.join(path, txt_name)
            with open(file_path, 'r') as f:
                info = f.readlines()
                infoes.append(info)

        savepath, _ = os.path.split(path)
        savepath = os.path.join(savepath , f"{mode}.txt")
        print(savepath)
        with open( savepath, 'w') as f:
            for info in infoes:
                for key in info:
                    f.write(key)



if __name__ == "__main__":
    txt_savepath = r'F:\Model_output\badcaseTXT'
    main(txt_savepath)
