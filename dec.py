import os
from pathlib import Path

import matplotlib.pyplot as plt


def getGtAreaAndRatio(label_dir):
    """
    得到不同尺度的gt框个数
    :params label_dir: label文件地址
    :return data_dict: {dict: 3}  3 x {'类别':{’area':[...]}, {'ratio':[...]}}.
    """
    data_dict = {}
    assert Path(label_dir).is_dir(), "label_dir is not exist"

    txts = os.listdir(label_dir)

    for txt in txts:
        with open(os.path.join(label_dir, txt)) as f:
            lines = f.readlines()

        for line in lines:
            temp = line.split()
            coor_list = list(map(lambda x: x, temp[1:]))
            area = float(coor_list[2]) * float(coor_list[3])

            ratio = round(float(coor_list[2]) / float(coor_list[3]), 2)

            if temp[0] not in data_dict:
                data_dict[temp[0]] = {}
                data_dict[temp[0]]["area"] = []
                data_dict[temp[0]]["ratio"] = []

            data_dict[temp[0]]["area"].append(area)
            data_dict[temp[0]]["ratio"].append(ratio)

    return data_dict


def getSMLGtNumByClass(data_dict, class_num):
    """
    计算某个类别的小物体、中物体、大物体的个数
    params data_dict: {dict: 3}  3 x {'类别':{’area':[...]}, {'ratio':[...]}}
    params class_num: 类别  0, 1, 2
    return s: 该类别小物体的个数  0 < area <= 0.5%
           m: 该类别中物体的个数  0.5% < area <= 1%
           l: 该类别大物体的个数  area > 1%.
    """
    s, m, l = 0, 0, 0
    # 图片的尺寸大小 注意修改!!!
    h = 640
    w = 640
    for item in data_dict[f"{class_num}"]["area"]:
        if item * h * w <= h * w * 0.005:
            s += 1
        elif item * h * w <= h * w * 0.010:
            m += 1
        else:
            l += 1
    return s, m, l


def getAllSMLGtNum(data_dict, isEachClass=False):
    """
    数据集所有类别小、中、大GT分布情况
    isEachClass 控制是否按每个类别输出结构.
    """
    S, M, L = 0, 0, 0
    # 需要手动初始化下，有多少个类别就需要写多个
    # classDict = {'0': {'S': 0, 'M': 0, 'L': 0}, '1': {'S': 0, 'M': 0, 'L': 0}, '2': {'S': 0, 'M': 0, 'L': 0}}
    classDict = {"0": {"S": 0, "M": 0, "L": 0}, "1": {"S": 0, "M": 0, "L": 0}}

    print(classDict["0"]["S"])
    # range(class_num)类别数 注意修改!!!
    if not isEachClass:
        for i in range(2):
            s, m, l = getSMLGtNumByClass(data_dict, i)
            S += s
            M += m
            L += l
        return [S, M, L]
    else:
        for i in range(1):
            S = 0
            M = 0
            L = 0
            s, m, l = getSMLGtNumByClass(data_dict, i)
            S += s
            M += m
            L += l
            classDict[str(i)]["S"] = S
            classDict[str(i)]["M"] = M
            classDict[str(i)]["L"] = L
        return classDict


def plotAllSML(SML):
    x = ["S:[0, 32x32]", "M:[32x32, 96x96]", "L:[96x96, 640x640]"]
    plt.figure(figsize=(10, 8))
    plt.bar(x, SML, width=0.5, align="center", color=["skyblue", "orange", "green"])
    for a, b, i in zip(x, SML, range(len(x))):
        plt.text(a, b + 0.01, "%d" % int(SML[i]), ha="center", fontsize=15, color="r")
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel("GT Size", fontsize=16)
    plt.ylabel("Numbers", fontsize=16)
    plt.title("Small, medium, and large GT distributions in the dataset", fontsize=16)
    plt.show()

    plt.savefig("111.png")


if __name__ == "__main__":
    labeldir = r"D:\googleDownload\pig_fight_DEC\train\labels"  # 数据集标签路径
    data_dict = getGtAreaAndRatio(labeldir)
    # 1、数据集所有类别小、中、大GT分布情况
    # 控制是否按每个类别输出结构
    isEachClass = True
    SML = getAllSMLGtNum(data_dict, isEachClass)
    print(SML)
    if not isEachClass:
        plotAllSML(SML)
