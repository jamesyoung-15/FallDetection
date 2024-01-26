from ultralytics import YOLO
import argparse


parser = argparse.ArgumentParser()

parser.add_argument("--src", type=str, default='0', help="Video file location (eg. /dev/video0)")

args = parser.parse_args()

media_src = args.src

model_path = "../models/yolo-weights/yolov8-fall.pt"
model = YOLO(model_path)

results = model(source=media_src, show=True, conf=0.5)