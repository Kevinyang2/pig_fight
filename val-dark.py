import os
import shutil

import cv2
import numpy as np

# 输入输出路径
input_images = "D:\googleDownload\pig_fight_DEC\\valid\images"
input_labels = "D:\googleDownload\pig_fight_DEC\\valid\labels"

output_images = "D:\googleDownload\\test_dark/images"
output_labels = "D:\googleDownload\\test_dark/labels"

os.makedirs(output_images, exist_ok=True)
os.makedirs(output_labels, exist_ok=True)


# Gamma 校正函数
def adjust_gamma(image, gamma=2.0):
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(image, table)


# 遍历处理图像
for filename in os.listdir(input_images):
    if filename.endswith((".jpg", ".png", ".jpeg")):
        # 读入图片
        img_path = os.path.join(input_images, filename)
        img = cv2.imread(img_path)

        # 变暗处理
        darker = adjust_gamma(img, gamma=5.0)

        # 可选：叠加噪声
        noise = np.random.normal(0, 15, img.shape).astype(np.int16)
        darker_noisy = np.clip(darker + noise, 0, 255).astype(np.uint8)

        # 保存新图像
        save_img_path = os.path.join(output_images, filename)
        cv2.imwrite(save_img_path, darker)

        # 复制对应的 label 文件
        label_name = os.path.splitext(filename)[0] + ".txt"
        src_label = os.path.join(input_labels, label_name)
        dst_label = os.path.join(output_labels, label_name)

        if os.path.exists(src_label):
            shutil.copy(src_label, dst_label)

print("✅ 昏暗版数据集生成完成，图像和标注已保存到:", output_images, output_labels)
