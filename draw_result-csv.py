import matplotlib.pyplot as plt
import pandas as pd

# 读取 CSV 文件
data = pd.read_csv("runs/train/v10-BS_exp/results.csv")  # 替换成训练结果的csv路径
data_2 = pd.read_csv("runs/train/v10-APConv_exp/results.csv")  # 替换成训练结果的csv路径
data_3 = pd.read_csv("runs/train/v10-ATFLm_exp/results.csv")  # 替换成训练结果的csv路径
data_4 = pd.read_csv("runs/train/v10-APConv+ATFLm_exp/results.csv")  # 替换成训练结果的csv路径
data_5 = pd.read_csv("runs/train/v10-APconv_AssemFormer-HSFPN_exp/results.csv")  # 替换成训练结果的csv路径

# 获取'metrics/mAP_0.5'列的数据
mAP_05_data = data["metrics/mAP50(B)"]
mAP_05_data_2 = data_2["metrics/mAP50(B)"]
mAP_05_data_3 = data_3["metrics/mAP50(B)"]
mAP_05_data_4 = data_4["metrics/mAP50(B)"]
mAP_05_data_5 = data_4["metrics/mAP50(B)"]

# 绘制曲线
plt.plot(mAP_05_data, label="v10-baseline", color="red", linewidth=1)
plt.plot(mAP_05_data_2, label="v10-AP", color="green", linewidth=1)
plt.plot(mAP_05_data_3, label="v10-ATFlm", color="blue", linewidth=1)
plt.plot(mAP_05_data_4, label="v10-AP-ATFlm", color="yellow", linewidth=1)
plt.plot(mAP_05_data_4, label="v10-Ap-ATFlm-AF-HSFPN", color="black", linewidth=1)

# 添加图例
plt.legend(loc="lower right")

# 添加标题和坐标轴标签
plt.xlabel("Epoch")
plt.ylabel("mAP50(B)")
plt.title("mAP50(B)Curve")

# 网格线
plt.grid(True)

# 保存图像到同目录下
plt.savefig("results/mAP50(B)impro.png")
plt.show()
