from ultralytics import YOLO
import argparse
import cv2
import time

par = argparse.ArgumentParser(description='Human Fall Detection Demo.')
par.add_argument('-v', '--video', default=0,  help='Source of camera or video file path. Eg. /dev/video0 or ./videos/myvideo.mp4')
par.add_argument('-s', '--stream', default=1, help='Specify whether stream or not. 0 for true 1 for false.')
par.add_argument('-o', '--show', default=1, help='Specify whether to show output or not. 0 for true 1 for false.')
args = par.parse_args()
print(args)
vid_source = args.video
is_stream = args.stream
to_show = args.show

# load pretrained model
model = YOLO("yolo-weights/yolov8n-pose.pt")

# # run predict
# results = model(source=args.video, show=bool(int(to_show)))
cap = cv2.VideoCapture(vid_source)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

prev_time = 0

KEYPOINT_DICT = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}

prev_data = [None]*2
print(prev_data)

interval = 1

while True:
    ret, frame = cap.read()
    
    curr_time = time.time()
    elapsed_time = curr_time - prev_time
    if elapsed_time >= interval:
        prev_time = curr_time
        results = model.predict(frame, imgsz=640, conf=0.5)
        for result in results:
            keypts = result.keypoints
            # print(f'Keypoints: \n{kpts}')
            num_people = keypts.shape[0]
            num_pts = keypts.shape[1]
            

            if num_pts !=0:
                for i in range(num_people):
                    left_shoulder = keypts.xy[i, KEYPOINT_DICT['left_shoulder']]
                    right_shoulder = keypts.xy[i, KEYPOINT_DICT['right_shoulder']]
                    left_hip = keypts.xy[i, KEYPOINT_DICT['left_hip']]
                    right_hip = keypts.xy[i, KEYPOINT_DICT['right_hip']]
                    left_knee = keypts.xy[i, KEYPOINT_DICT['left_knee']]
                    right_knee = keypts.xy[i, KEYPOINT_DICT['right_knee']]
                    for index in KEYPOINT_DICT.values():
                        keypoint = keypts.xy[i, index]
                        x, y = int(keypoint[0].item()), int(keypoint[1].item())
                        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
    
    cv2.imshow('Yolo Pose Test', frame)
    key = cv2.waitKey(1)
    if key == 27:  # ESC
        break
cap.release()
cv2.destroyAllWindows()