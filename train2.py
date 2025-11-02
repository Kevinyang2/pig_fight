import warnings

warnings.filterwarnings("ignore")
from ultralytics import RTDETR, YOLO

if __name__ == "__main__":
    model = YOLO(model=RTDETR(r"D:\ultralytics\ultralytics\cfg\models\11\yolo11-RTDETRDecoder.yaml"))
    # model.load('yolo12n.pt') # 加载预训练权重,改进或者做对比实验时候不建议打开，因为用预训练模型整体精度没有很明显的提升
    model.train(
        data=r"pig.yaml",
        imgsz=640,
        epochs=200,
        batch=16,
        workers=6,
        device="0",
        optimizer="SGD",
        patience=20,
        augment=False,
        close_mosaic=10,
        resume=False,
        project="runs/train",
        name="v11-RTDETRDecoder_exp",
        cache=False,
    )
