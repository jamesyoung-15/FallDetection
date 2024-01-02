from ultralytics import YOLO
import argparse

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

# run predict
results = model(source=args.video, show=bool(int(to_show)))
