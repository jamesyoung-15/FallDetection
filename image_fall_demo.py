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
parser.add_argument("--delay", type=int, default=1, help="CV2 show waitkey delay between frames. Default is 1, change to higher if fps is too high")
parser.add_argument("--resize", type=int, default=0, help="Whether to resize image or not. Default 0 means no resize, 1 is resize")
parser.add_argument("--width", type=int, default=640, help="Resize video width. (eg. 640)")
parser.add_argument("--height", type=int, default=480, help="Resize video height (eg. 480)")
# can always add more args here
args = parser.parse_args()

delay = args.delay
vid_source = args.src
interval = args.interval
resize = bool(args.resize)
resize_width = args.width
resize_height = args.height

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
    
    # Optional: resize frame (can increase fps if downsize resolution at cost of accuracy)
    if resize:
        image = cv2.resize(image, (resize_width,resize_height))
    
    # inference
    if frame_counter >= interval:
        frame_counter = 0
        results = model.predict(image, conf=0.5, verbose=False)
        # extract class from results
        for result in results:
            # draw results to image
            image = result.plot()
            for c in result.boxes.cls:
                # here we can do something with the class
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
    # press q to exit
    if cv2.waitKey(delay) == ord('q'):
        break

# cleanup
cap.release()
cv2.destroyAllWindows()
