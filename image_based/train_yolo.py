from ultralytics import YOLO

model_path = "../models/yolo-weights/yolov8n.pt"
model = YOLO(model_path)


results = model.train(data='fall-dataset.yaml', epochs=300, imgsz=640)
# results = model.train(data='fall-dataset-2.yaml', epochs=300, imgsz=640)