from ultralytics import YOLO

# Load a model
# model = YOLO("yolo11n.pt")  # load an official model
model = YOLO("runs/train/v10-Test_exp7/weights/best.pt")  # load a custom model

# Predict with the model
results = model("F:\pig_fight_DEC\\valid\images\\000001_jpg.rf.50f7ead7b6fff2f8d38b959c0c35f0d1.jpg",
                show=True,
                save=True,
                save_txt=False,
                visualize=False)  # predict on an image

# Access the results
for result in results:
    xywh = result.boxes.xywh  # center-x, center-y, width, height
    xywhn = result.boxes.xywhn  # normalized
    xyxy = result.boxes.xyxy  # top-left-x, top-left-y, bottom-right-x, bottom-right-y
    xyxyn = result.boxes.xyxyn  # normalized
    names = [result.names[cls.item()] for cls in result.boxes.cls.int()]  # class name of each box
    confs = result.boxes.conf  # confidence score of each box


##“F:\pig_fight_DEC\\valid\images\\000003_jpg.rf.6b7fa523138db2642d2c69b3222e678d.jpg”  大幅度动作识别不准
##"F:\pig_fight_DEC\\valid\images\\000003_jpg.rf.e47aee7a549f04c5372b7ca8c9553180.jpg" 小动作相似识别分类错误