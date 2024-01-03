from ultralytics import YOLO
import argparse
import cv2

par = argparse.ArgumentParser(description='Human Fall Detection Demo.')
par.add_argument('-i', '--image', default=None,  help='Source of camera or video file path. Eg. /dev/video0 or ./videos/myvideo.mp4')
args = par.parse_args()
print(args)
img_src = args.image

if img_src is None:
    raise Exception("No image specified")

# load pretrained model
model = YOLO("yolo-weights/yolov8n-pose.pt")

# # run predict
# results = model(source=args.video, show=bool(int(to_show)))
frame = cv2.imread(img_src)
frame = cv2.resize(frame, (640,640))

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
# want shoulder, hip, and knee
important_points = [5,6,11,12,13,14]

results = model.predict(frame, conf=0.5)
for result in results:
    keypts = result.keypoints
    # print(f'Keypoints: \n{kpts}')
    num_people = keypts.shape[0]
    num_pts = keypts.shape[1]
    
    if num_pts !=0:
        for i in range(num_people):
            for index in important_points.values():
                keypoint = keypts.xy[i, index]
                x, y = int(keypoint[0].item()), int(keypoint[1].item())
                print(x,y)
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)


cv2.imshow('Yolo Pose Test', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()