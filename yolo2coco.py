import argparse
import json
import os

import cv2
from tqdm import tqdm

classes = ["pig_fight", "pig_normal"]

parser = argparse.ArgumentParser()
parser.add_argument(
    "--image_path", default="D:\googleDownload\pig_fight_DEC\\valid\images", type=str, help="path of images"
)
parser.add_argument(
    "--label_path", default="D:\googleDownload\pig_fight_DEC\\valid\labels", type=str, help="path of labels .txt"
)
parser.add_argument(
    "--save_path", default="data.json", type=str, help="if not split the dataset, give a path to a json file"
)
arg = parser.parse_args()


def yolo2coco(arg):
    print("Loading data from ", arg.image_path, arg.label_path)

    assert os.path.exists(arg.image_path)
    assert os.path.exists(arg.label_path)

    originImagesDir = arg.image_path
    originLabelsDir = arg.label_path
    # images dir name
    indexes = os.listdir(originImagesDir)

    dataset = {"categories": [], "annotations": [], "images": []}
    # 使用 1-based 的 category_id（1..nc），与验证导出的 predictions.json 对齐
    for i, cls in enumerate(classes, 1):
        dataset["categories"].append({"id": i, "name": cls, "supercategory": "mark"})

    ann_id_cnt = 0
    for k, index in enumerate(tqdm(indexes)):
        txtFile = f"{index[: index.rfind('.')]}.txt"
        stem = index[: index.rfind(".")]
        try:
            im = cv2.imread(os.path.join(originImagesDir, index))
            height, width, _ = im.shape
        except Exception as e:
            print(f"{os.path.join(originImagesDir, index)} read error.\nerror:{e}")
        if not os.path.exists(os.path.join(originLabelsDir, txtFile)):
            continue
        dataset["images"].append({"file_name": index, "id": stem, "width": width, "height": height})
        with open(os.path.join(originLabelsDir, txtFile)) as fr:
            labelList = fr.readlines()
            for label in labelList:
                label = label.strip().split()
                x = float(label[1])
                y = float(label[2])
                w = float(label[3])
                h = float(label[4])

                H, W, _ = im.shape
                x1 = (x - w / 2) * W
                y1 = (y - h / 2) * H
                x2 = (x + w / 2) * W
                y2 = (y + h / 2) * H
                cls_id = int(label[0])
                width = max(0, x2 - x1)
                height = max(0, y2 - y1)
                dataset["annotations"].append(
                    {
                        "area": width * height,
                        "bbox": [x1, y1, width, height],
                        # 将 YOLO 的 0-based 标签转换为 1-based 的 COCO category_id
                        "category_id": cls_id + 1,
                        "id": ann_id_cnt,
                        "image_id": stem,
                        "iscrowd": 0,
                        "segmentation": [[x1, y1, x2, y1, x2, y2, x1, y2]],
                    }
                )
                ann_id_cnt += 1

    with open(arg.save_path, "w") as f:
        json.dump(dataset, f)
        print(f"Save annotation to {arg.save_path}")


if __name__ == "__main__":
    yolo2coco(arg)
