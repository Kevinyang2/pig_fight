from ultralytics import YOLO

model = YOLO("runs/train/v10-APConv-AssemFormer-HSFPN-ATFLm_exp/weights/best.pt")
model.export(format="onnx", opset=17, imgsz=640, dynamic=False)
