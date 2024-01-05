from ultralytics import YOLO
import utils

args = utils.get_args()

media_src = args.src

model_path = "./yolo-weights/yolov8-fall.pt"
model = YOLO(model_path)

results = model(source=media_src, show=True)