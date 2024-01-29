from ultralytics import YOLO
import argparse
import cv2

class_dict = {0: 'falling', 1: 'sitting', 2:'standing', 3:'walking'}

parser = argparse.ArgumentParser()

parser.add_argument("--src", type=str, default='0', help="Video file location (eg. ./video.mp4)")

args = parser.parse_args()

vid_source = args.src

model_path = "./models/yolo-weights/yolov8-fall2.pt"
model = YOLO(model_path)

# results = model(source=media_src, show=True, conf=0.55)

cap = cv2.VideoCapture(vid_source)
while cap.isOpened():
    success, image = cap.read()
    if not success:
        break
    results = model.predict(image, conf=0.5, verbose=False)
    for result in results:
        image = result.plot()
        for c in result.boxes.cls:
            state = class_dict[int(c)]
            print(state)
            if state == 'falling':
                print('Fall Detected!')
    cv2.imshow('Yolo Fall Detection', image)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
