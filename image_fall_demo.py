from ultralytics import YOLO
import argparse
import cv2
import time

# defs
class_dict = {0: 'falling', 1: 'sitting', 2:'standing', 3:'walking'}

# get user passed args
parser = argparse.ArgumentParser()
parser.add_argument("--src", type=str, default='0', help="Video file location (eg. ./video.mp4)")
parser.add_argument("--interval", type=int, default=5, help="Frame interval between inference")
args = parser.parse_args()

vid_source = args.src
interval = args.interval

# load Yolo model
model_path = "./models/yolo-weights/yolov8-fall.pt"
model = YOLO(model_path)
# results = model(source=media_src, show=True, conf=0.55)

# init variables
prev_time = 0
frame_counter = 0

cap = cv2.VideoCapture(vid_source)
while cap.isOpened():
    success, image = cap.read()
    if not success:
        break
    
    frame_counter += 1
    
    image = cv2.resize(image, (480,320))
    
    # inference
    if frame_counter >= interval:
        frame_counter = 0
        results = model.predict(image, conf=0.5, verbose=False)
        # extract class from results
        for result in results:
            image = result.plot()
            for c in result.boxes.cls:
                state = class_dict[int(c)]
                print(state)
                if state == 'falling':
                    print('Fall Detected!')
    
    # fps
    curr_time = time.time()
    fps = int(1/(curr_time-prev_time))
    prev_time = curr_time
    cv2.putText(image, f'FPS: {fps}', (10,30), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)
    
    # display    
    cv2.imshow('Yolo Fall Detection', image)
    if cv2.waitKey(1) == ord('q'):
        break

# cleanup
cap.release()
cv2.destroyAllWindows()
