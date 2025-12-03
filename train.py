# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
import pdb

if __name__ == '__main__':
    model = YOLO(model=r'ultralytics/cfg/models/v8/yolov8_APconv.yaml')
    # model.load('yolo11n.pt') # 加载预训练权重,改进或者做对比实验时候不建议打开，因为用预训练模型整体精度没有很明显的提升
    model.train(data=r'strawberries.yaml',
                imgsz=640,
                epochs=300,
                batch=128,
                workers=10,
                device='0',
                optimizer='SGD',
                patience= 30,
                augment=False,
                close_mosaic=10,
                lr0=0.001,
                resume=False,
                project='runs/train',
                name='v8-APconv_strawberry',
               
    )
