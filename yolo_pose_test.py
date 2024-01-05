# import the necessary packages
from ultralytics import YOLO
import cv2
import time
import utils
import my_defs
from camera_stream.usbcam_stream import USBCamStream

def get_xy(keypoint):
    """ Convert Yolo tensor keypoint data to array and returns (x,y)  """
    try:
        return int(keypoint[0].item()), int(keypoint[1].item())
    except:
        raise Exception("unable to get keypoint coordinate")

def get_conf(conf_scores, part):
    """ Return confidence score for each keypoint (float). Input conf_scores is keypoint.conf (list). """
    try:
        return float(conf_scores[my_defs.KEYPOINT_DICT[part]])
    except:
        raise Exception("unable to get confidence score")
    

def main():
    # get user passed args
    args = utils.get_args()
    vid_source = args.src
    show_frame = args.show
    manual_move = bool(int(args.manual_frame))
    conf_threshold = args.conf_score
    vid_width = args.width
    vid_height = args.height
    
    # load pretrained model
    model = YOLO("yolo-weights/yolov8n-pose.pt")

    # load video src
    # cap = cv2.VideoCapture(vid_source)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, vid_width)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, vid_height)
    cap = USBCamStream(src=vid_source)
    cap.resize_stream(vid_width,vid_height)
    cap = cap.start()
    cap.change_format()

    # array to store prev frame data for determining action
    prev_data = [None]

    # time variables
    prev_time = 0 # for tracking interval to execute predict with Yolo model
    interval = args.interval # interval to execute predict with Yolo model
    prev_time_fps = 0 # for fps counting

    while True:
        # read frame
        ret, frame = cap.read()
        if not ret:
            break
        
        # track time for interval and fps
        curr_time = time.time()
        elapsed_time = curr_time - prev_time
        
        # if specified predict interval
        if elapsed_time >= interval:
            # update timer
            prev_time = curr_time
            
            # inference
            # results = model.predict(frame, imgsz=640, conf=0.5, verbose=False)
            results = model.track(frame, imgsz=640, conf=0.5, verbose=False, tracker="bytetrack.yaml")
            # get data from inference
            for result in results:
                keypts = result.keypoints
                # print(f'Keypoints: \n{kpts}')
                num_people = keypts.shape[0]
                num_pts = keypts.shape[1]
                
                
                # if keypts detected
                if num_pts !=0:
                    # for each person
                    for i in range(num_people):
                        # extract relevant keypts
                        left_shoulder = get_xy(keypts.xy[i, my_defs.KEYPOINT_DICT['left_shoulder']])
                        right_shoulder = get_xy(keypts.xy[i, my_defs.KEYPOINT_DICT['right_shoulder']])
                        left_hip = get_xy(keypts.xy[i, my_defs.KEYPOINT_DICT['left_hip']])
                        right_hip = get_xy(keypts.xy[i, my_defs.KEYPOINT_DICT['right_hip']])
                        left_knee = get_xy(keypts.xy[i, my_defs.KEYPOINT_DICT['left_knee']])
                        right_knee = get_xy(keypts.xy[i, my_defs.KEYPOINT_DICT['right_knee']])
                        # get confidence scores
                        left_shoulder_conf = get_conf(keypts.conf[i], 'left_shoulder')
                        right_shoulder_conf = get_conf(keypts.conf[i], 'right_shoulder')
                        left_hip_conf = get_conf(keypts.conf[i], 'left_hip')
                        right_hip_conf = get_conf(keypts.conf[i], 'right_hip')
                        left_knee_conf = get_conf(keypts.conf[i], 'left_knee')
                        right_knee_conf = get_conf(keypts.conf[i], 'right_knee')
                        
                        # check whether left/right keypt exist, if both exist get midpoint
                        shoulder = utils.get_mainpoint(left_shoulder, right_shoulder, left_shoulder_conf, right_shoulder_conf, conf_threshold=conf_threshold,  part = "shoulders")
                        hips = utils.get_mainpoint(left_hip, right_hip, left_hip_conf, right_hip_conf, conf_threshold=conf_threshold, part = "hips")
                        knees = utils.get_mainpoint(left_knee, right_knee, left_knee_conf, right_knee_conf, conf_threshold=conf_threshold, part = "knees")
                        shoulder_exist = False
                        hips_exist = False
                        knees_exist = False
                        
                        # if relevant keypt exist draw pt
                        if shoulder!=(0,0):
                            shoulder_exist = True
                            utils.draw_keypoint(frame, shoulder)
                            # print(f'Shoulder: {shoulder}')
                        if hips!=(0,0):
                            hips_exist = True
                            utils.draw_keypoint(frame, hips)
                            # print(f'Hips: {hips}')
                        if knees!=(0,0):
                            knees_exist = True
                            utils.draw_keypoint(frame, knees)
                            # print(f'Knees: {knees}')
                            
                        # if keypts exist draw line to connect them, calculate vector
                        spine_vector = (0,0)
                        legs_vector = (0,0)
                        if shoulder_exist and hips_exist:
                            spine_vector = utils.calculate_vector(hips, shoulder)
                            # utils.draw_keypoint_line(frame, shoulder, hips)
                            utils.draw_vector(frame, hips, spine_vector)
                            # print(f'Spine Vector: {spine_vector}')
                        if hips_exist and knees_exist:
                            legs_vector = utils.calculate_vector(hips, knees)
                            # utils.draw_keypoint_line(frame, hips, knees)
                            utils.draw_vector(frame, hips, legs_vector)
                            # print(f'Leg Vector: {legs_vector}')
                        
                        # calculate vector if all 3 main pts exist
                        spine_leg_theta = -1 # angle between spine (vector between shoulder and hips) and legs (vector between hips and knees)
                        spine_x_axis_phi = -1 # angle between spine (vector between shoulder and hips) and x_axis along hip point
                        standing = None
                        sitting = None
                        lying_down = None
                        if shoulder_exist and hips_exist and knees_exist:
                            spine_leg_theta = utils.angle_between(spine_vector, legs_vector)
                            hips_x_axis = utils.calculate_vector(hips, (hips[0]+20, hips[1]))
                            hips_y_axis = utils.calculate_vector(hips, (hips[0], hips[1]+20))
                            # utils.draw_vector(frame, hips, hips_x_axis, color=(255,255,255))
                            # spine_x_axis_phi = utils.calculate_angle_with_x_axis(spine_vector)
                            spine_x_axis_phi = utils.angle_between(spine_vector, hips_x_axis)
                            legs_y_axis_alpha = utils.angle_between(legs_vector, hips_y_axis)
                            print(f'Person {i+1}')
                            print(f'Theta {spine_leg_theta}, Phi: {spine_x_axis_phi}, Alpha: {legs_y_axis_alpha}')
                            state = utils.action_state(spine_leg_theta, spine_x_axis_phi, legs_y_axis_alpha)
                            print(f'State: {state}')

                            
        # track fps and draw to frame
        fps = 1/(curr_time-prev_time_fps)
        cv2.putText(frame, str(int(fps)), (50,50),  cv2.FONT_HERSHEY_PLAIN,3,(225,0,0),3)
        
        if bool(int(show_frame)):
            cv2.imshow('Yolo Pose Test', frame)
        prev_time_fps = curr_time
        key = cv2.waitKey(1)
        if manual_move:
            key = cv2.waitKey(0)
            if key == ord("n"):
                continue
        if key == 27:  # ESC
            break
        
    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main()