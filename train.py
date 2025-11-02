# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
import pdb

if __name__ == '__main__':
    model = YOLO(model=r'D:\ultralytics\ultralytics\cfg\models\v10\yolov10n-multiframe.yaml')
    # model.load('yolo12n.pt') # 加载预训练权重,改进或者做对比实验时候不建议打开，因为用预训练模型整体精度没有很明显的提升
    model.train(data=r'pig.yaml',
                imgsz=640,
                epochs=200,
                batch=32,
                workers=6,
                device='0',
                optimizer='SGD',
                patience= 30,
                augment=False,
                close_mosaic=10,
                # lr0=0.001,
                resume=False,
                project='runs/train',
                name='v10-Test_exp',
                cache=False,
                t_frames=4,
                frame_stride=1,
                hsv_h=0,
                hsv_s = 0,
                hsv_v = 0,
                )
