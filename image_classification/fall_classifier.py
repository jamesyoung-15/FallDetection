from ultralytics import YOLO
from utils import pose_utils

args = pose_utils.get_args()

media_src = args.src

model_path = "../models/yolo-weights/yolov8-fall.pt"
model = YOLO(model_path)

results = model(source=media_src, show=True, conf=0.5)