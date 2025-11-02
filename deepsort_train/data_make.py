import os

import cv2


def main():
    # 图像文件夹路径
    img_path = "G:\pig.v10i.yolov11\\test\images\\"
    # TXT格式的标注文件夹路径
    anno_path = "G:\pig.v10i.yolov11\\test\labels\\"
    # 裁剪图像输出路径
    cut_path = "G:\\ultralytics\deepsort_train\data\\test"

    if not os.path.exists(cut_path):
        os.makedirs(cut_path)

    imagelist = os.listdir(img_path)

    for image in imagelist:
        image_pre, _ext = os.path.splitext(image)
        img_file = os.path.join(img_path, image)
        txt_file = os.path.join(anno_path, image_pre + ".txt")

        if not os.path.exists(txt_file):
            continue

        img = cv2.imread(img_file)
        if img is None:
            continue

        height, width = img.shape[:2]
        with open(txt_file) as f:
            lines = f.readlines()

        obj_i = 0
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                continue

            cls_id, x_center, y_center, w, h = map(float, parts)
            x_center *= width
            y_center *= height
            w *= width
            h *= height

            xmin = int(x_center - w / 2)
            ymin = int(y_center - h / 2)
            xmax = int(x_center + w / 2)
            ymax = int(y_center + h / 2)

            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(width - 1, xmax)
            ymax = min(height - 1, ymax)

            img_cut = img[ymin:ymax, xmin:xmax, :]
            class_name = str(int(cls_id))  # 你也可以使用类别名映射表

            path = os.path.join(cut_path, class_name)
            os.makedirs(path, exist_ok=True)

            obj_i += 1
            try:
                save_path = os.path.join(path, f"{image_pre}_{obj_i:0>2d}.jpg")
                cv2.imwrite(save_path, img_cut)
            except:
                continue

            print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()
