import pandas as pd
import matplotlib.pyplot as plt

# 读取 CSV 文件
data = pd.read_csv('runs/train/strawberry/rt-detr-resnet50_strawberry/results.csv')  #替换成训练结果的csv路径
data_2 = pd.read_csv('runs/train/strawberry/v8_strawberry/results.csv') #替换成训练结果的csv路径
data_3 = pd.read_csv('runs/train/strawberry/v9t_strawberry/results.csv') #替换成训练结果的csv路径
data_4 = pd.read_csv('runs/train/strawberry/v10_strawberry/results.csv') #替换成训练结果的csv路径
data_5 = pd.read_csv('runs/train/strawberry/v11-strawberry/results.csv') #替换成训练结果的csv路径
data_6 = pd.read_csv('runs/train/strawberry/v12-strawberry/results.csv') #替换成训练结果的csv路径

# 获取'metrics/mAP_0.5'列的数据
mAP_05_data = data['metrics/mAP50(B)']
mAP_05_data_2 = data_2['metrics/mAP50(B)']
mAP_05_data_3 = data_3['metrics/mAP50(B)']
mAP_05_data_4 = data_4['metrics/mAP50(B)']
mAP_05_data_5 = data_5['metrics/mAP50(B)']
mAP_05_data_6 = data_6['metrics/mAP50(B)']

# 绘制曲线
plt.plot(mAP_05_data, label='rt-detr-resnet50', color='red', linewidth=1)
plt.plot(mAP_05_data_2, label='v8', color='green', linewidth=1)
plt.plot(mAP_05_data_3, label='v9', color='blue', linewidth=1)
plt.plot(mAP_05_data_4, label='v10', color='yellow', linewidth=1)
plt.plot(mAP_05_data_5, label='v11', color='black', linewidth=1)
plt.plot(mAP_05_data_6, label='v12', color='Purple', linewidth=1)

# 添加图例
plt.legend(loc='lower right')

# 添加标题和坐标轴标签
plt.xlabel('Epoch')
plt.ylabel('mAP50(B)')
plt.title('mAP50(B)Curve')

# 网格线
plt.grid(True)

# 保存图像到同目录下
plt.savefig('results/mAP50(B)impro.png')
plt.show()
