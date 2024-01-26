import cv2
from ultralytics import YOLO
from Pose_Estimate.pose_fall_detection import PoseFallDetector
from Pose_Estimate.yolo_pose.yolo_pose import YoloPoseDetector
from utils import pose_utils
import time
from movenet_pose_fall_detect import run as movenet_run
from yolo_pose_fall_detect import run as yolo_run


def main():
    # get user passed args
    args = pose_utils.get_args()
    media_source = args.src
    show_frame = args.show
    manual_move = bool(int(args.manual_frame))
    conf_threshold = args.conf_score
    vid_width = args.width
    vid_height = args.height
    is_image = bool(int(args.type))
    interval = args.interval
    debug = bool(args.debug)
    save_video = bool(args.save_vid)
    pose_model = args.pose_type
    resize = bool(args.resize_frame)
    delay = args.delay
    fps = args.fps
    benchmark = bool(args.benchmark)
    
    # load model
    model = None
    if pose_model==1:
        print("Using Movenet Multipose Lightning")
        movenet_run(estimation_model='models/tflite/movenet_multipose.tflite', tracker_type="bounding_box",
                    media_src=media_source, vid_width=vid_width, vid_height=vid_height, delay=delay, fps=fps, 
                    save_video=save_video, benchmark=benchmark, resize=resize)
    else:
        print("Using YoloV8 Pose")
        yolo_run(media_source, vid_width=vid_width, vid_height=vid_height, interval=interval, 
                        debug=debug, save_video=save_video, conf_threshold=conf_threshold, 
                        resize=resize, delay=delay, fps=fps, benchmark=benchmark)


if __name__ == "__main__":
    main()