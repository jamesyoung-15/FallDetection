from ultralytics import YOLO
import argparse
import cv2
import time
import utils
import my_defs


def get_xy(keypoint):
    try:
        return int(keypoint[0].item()), int(keypoint[1].item())
    except:
        raise Exception("unable to get keypoint coordinate")

def draw_keypoint(image, keypoint):
    image = cv2.circle(image, (keypoint[0], keypoint[1]), 5, (0, 255, 0), -1)
    return image

def get_mainpoint(left, right, part):
    """ For each important part (eg. hip), if both left and right part exists, we get midpoint. If only left or right exist, then we set the point to left or right. """
    main_point = (0,0)
    if left != (0,0) and right != (0,0):
        print(f'both left and right {part} detected')
        main_point = utils.calculate_midpoint(left, right)
    elif left != (0,0):
        print(f'only left {part} detected')
        main_point = left
    elif right != (0,0):
        print(f'only right {part} detected')
        shoulder = right
    return main_point

def main():
    args = utils.get_args()
    vid_source = args.src
    # load pretrained model
    model = YOLO("yolo-weights/yolov8n-pose.pt")

    # # run predict
    # results = model(source=args.video, show=bool(int(to_show)))
    cap = cv2.VideoCapture(vid_source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

    prev_data = [None]

    prev_time = 0
    interval = args.interval
    prev_time_fps = 0

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
                    for i in range(1):
                        left_shoulder = get_xy(keypts.xy[i, my_defs.KEYPOINT_DICT['left_shoulder']])
                        right_shoulder = get_xy(keypts.xy[i, my_defs.KEYPOINT_DICT['right_shoulder']])
                        left_hip = get_xy(keypts.xy[i, my_defs.KEYPOINT_DICT['left_hip']])
                        right_hip = get_xy(keypts.xy[i, my_defs.KEYPOINT_DICT['right_hip']])
                        left_knee = get_xy(keypts.xy[i, my_defs.KEYPOINT_DICT['left_knee']])
                        right_knee = get_xy(keypts.xy[i, my_defs.KEYPOINT_DICT['right_knee']])
                        
                        shoulder = get_mainpoint(left_shoulder, right_shoulder, "shoulder")
                        hips = get_mainpoint(left_hip, right_hip, "hips")
                        knees = get_mainpoint(left_knee, right_knee, "knees")
                        if shoulder!=(0,0):
                            frame = draw_keypoint(frame, shoulder)
                        if hips!=(0,0):
                            frame = draw_keypoint(frame, hips)
                        if knees!=(0,0):
                            frame = draw_keypoint(frame, knees)
                        
                        
                    

                        # for index in KEYPOINT_DICT.values():
                        #     keypoint = keypts.xy[i, index]
                        #     x, y = int(keypoint[0].item()), int(keypoint[1].item())
                        #     cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                            
        
        fps = 1/(curr_time-prev_time_fps)
        cv2.putText(frame, str(int(fps)), (50,50),  cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)
        cv2.imshow('Yolo Pose Test', frame)
        prev_time_fps = curr_time
        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break
    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main()