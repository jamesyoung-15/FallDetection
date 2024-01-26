import cv2
from ultralytics import YOLO
from Pose_Estimate.pose_fall_detection import PoseFallDetector
from Pose_Estimate.yolo_pose.yolo_pose import YoloPoseDetector
from utils import pose_utils
import time

def video_inference(vid_source: str, vid_width: int, vid_height: int, 
                    show_frame: bool = True, manual_move: bool = False, interval: int = 5, 
                    debug: bool = False, save_video: bool = False, conf_threshold: float = 0.5, 
                    delay: int = 1,resize: bool = False, benchmark: bool = False, fps: int = 24):
    """ 
    Performs pose estimation and fall detection on video (can be usb camera or video file).
    
    """
    
    
    model = YoloPoseDetector(conf_threshold=conf_threshold, debug=debug, model_path="models/yolo-weights/yolov8n-pose.pt")
    model.load_model()
    fall_detector = PoseFallDetector(debug=debug)
    
    is_webcam = False
    if '/dev/video' in vid_source:
        is_webcam = True
    
    # non threaded video stream
    cap = cv2.VideoCapture(vid_source)
    # if video is webcam, set width and height, otherwise resizing with cap.set doesn't work
    if is_webcam:
        print("Detected using webcam...")
        print(f'Width: {vid_width}, Height: {vid_height}, FPS: {fps}')
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, vid_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, vid_height)
        cap.set(cv2.CAP_PROP_FPS, fps)
    elif not is_webcam and not resize:
        print("Detected non-webcam video...")
        vid_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # get width and height
    print(f'Video Width: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)} , Video Height: {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}')
    
    # save video
    if save_video == True:
        fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
        video_output = cv2.VideoWriter('demo.avi',  fourcc, 30, (int(cap.get(3)),int(cap.get(4)))) 

    # dictionary to store prev frame data for determining action
    prev_data = {}
    # other variables
    num_frames_elapsed = 0
    fall_detected = False
    fall_conf = 0
    prev_time = 0
    total_frames = 0
    start_time = time.time()
    fps_start_time = time.time()
    
    while True:
        # read frame
        ret, frame = cap.read()
        if not ret:
            break
        if resize:
            frame = cv2.resize(frame, (vid_width,vid_height), interpolation=cv2.INTER_AREA)
        # track time for checking time to delete old data
        curr_time = time.time()
        num_frames_elapsed += 1
        total_frames += 1
        # run inference every interval frames
        if num_frames_elapsed >= interval:
            num_frames_elapsed = 0
            model.yolo_track(prev_data, frame, curr_time)
            # fall_detected, fall_conf = fall_detector.fall_detection(prev_data, frame_width=vid_width, frame_height=vid_height)
            fall_detected, fall_conf = fall_detector.fall_detection(prev_data, frame_width=vid_width, frame_height=vid_height)
            
            
        
        # write to frame if fall detected
        if fall_detected and fall_conf>=0.7:
            cv2.putText(frame, "Fall Detected", (20,30),  cv2.FONT_HERSHEY_PLAIN,2,(0,0,245),3)

        # write person state (ie. sitting, standing, lying down) to frame
        for prev_data_key, prev_data_value in prev_data.copy().items():
            try:
                draw_point = prev_data[prev_data_key]['hips'][-1]
                temp_state = prev_data[prev_data_key]['state'][-1]
                temp_text = "ID " + str(prev_data_key) + ": " + temp_state
                cv2.putText(frame, temp_text, (draw_point[0]+10, draw_point[1]+20),  cv2.FONT_HERSHEY_PLAIN,1.5,(155,200,0),2)
            except:
                pass
        
        # save to video
        if save_video == True:
            video_output.write(frame)
        
        # fps track
        fps_frame_count = 5
        if total_frames % fps_frame_count == 0:
            fps_end_time = time.time()
            fps = fps_frame_count = 5/(fps_end_time-fps_start_time)
            fps_start_time = time.time()
            cv2.putText(frame, str(int(fps)), (50,50),  cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)
        
        # show frame to screen
        if show_frame:
            cv2.imshow('Yolo Pose Test', frame)
            
        # wait for user key
        key = cv2.waitKey(delay)
        if manual_move:
            key = cv2.waitKey(0)
        # press esc to quit
        if key == 27:  # ESC
            break
    
    # benchmark
    if benchmark:
        end_time = time.time()
        total_time = end_time - start_time
        total_fps = total_frames/(total_time)
        print(f'Total Frames: {total_frames}, Total Time: {total_time}, Total FPS: {total_fps}')
    
    # cleanup
    cap.release()
    if save_video == True:
        video_output.release()
    cv2.destroyAllWindows()

def image_inference(img_src, model):
    """ Performs pose estimation on image """
    frame = cv2.imread(img_src)
    frame = cv2.resize(frame, (640,640))
    prev_data = {}
    curr_time = 0
    model.yolo_predict(prev_data, frame, curr_time)
    # write person state (ie. sitting, standing, lying down) to frame
    for prev_data_key, prev_data_value in prev_data.copy().items():
        try:
            draw_point = prev_data[prev_data_key]['hips'][-1]
            temp_state = prev_data[prev_data_key]['state'][-1]
            temp_text = "ID " + str(prev_data_key) + ": " + temp_state
            cv2.putText(frame, temp_text, (draw_point[0]+10, draw_point[1]+20),  cv2.FONT_HERSHEY_PLAIN,1.5,(155,200,0),2)
        except:
            pass
    cv2.imshow('Yolo Pose Test', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def run(media_source, vid_width=640, vid_height=480, interval=5,conf_threshold =0.5, manual_move=False, is_image=False, benchmark=False, debug=False, show_frame=True, save_video=False, resize=False, delay=1, fps=24):
    if is_image:
        image_inference(img_src=media_source)
    else:
        video_inference(vid_source=media_source, vid_width=vid_width, vid_height=vid_height, 
                        show_frame=show_frame, manual_move=manual_move, interval=interval, 
                        debug=debug, save_video=save_video, conf_threshold=conf_threshold, 
                        resize=resize, delay=delay, fps=fps, benchmark=benchmark)

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
    # model = None
    # if pose_model==1:
    #     print("Using Movenet Multipose Lightning")
    # elif pose_model==2:
    #     print("Using Movenet Singlepose Lightning")
    # elif pose_model==3:
    #     print("Using Mediapipe Pose (single)")
    # else:
    
    print("Using YoloV8 Pose")
    model = YoloPoseDetector(conf_threshold=conf_threshold, debug=debug, model_path="models/yolo-weights/yolov8n-pose.pt")
    model.load_model()
    
    
    if is_image:
        image_inference(img_src=media_source, model=model)
    else:
        video_inference(vid_source=media_source, model=model, vid_width=vid_width, vid_height=vid_height, 
                        show_frame=show_frame, manual_move=manual_move, interval=interval, 
                        debug=debug, save_video=save_video, conf_threshold=conf_threshold, 
                        resize=resize, delay=delay, fps=fps, benchmark=benchmark)



if __name__ == "__main__":
    main()