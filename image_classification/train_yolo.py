from ultralytics import YOLO

model_path = "../models/yolo-weights/yolov8n.pt"
model = YOLO(model_path)


results = model.train(data='fall-dataset.yaml', epochs=20, imgsz=640)