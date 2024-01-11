# import the necessary packages
from ultralytics import YOLO
import cv2
import time
import utils
import my_defs
import numpy as np
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
    

def extract_keypts(person_num, keypts, frame, conf_threshold=0.5, track_id=0, debug=False):
    """ Performs inference on each person in frame. """
    # extract relevant keypoints and confidence scores into nested dict
    keypts_dict = {}
    for parts in my_defs.IMPORTANT_PTS:
        keypts_dict[parts] = {}
        keypts_dict[parts]['xy'] = get_xy(keypts.xy[person_num, my_defs.KEYPOINT_DICT[parts]])
        keypts_dict[parts]['conf_score'] = get_conf(keypts.conf[person_num], parts)
    
    # check whether left/right keypt exist, if both exist get midpoint
    shoulder = utils.get_mainpoint(keypts_dict['left_shoulder']['xy'], keypts_dict['right_shoulder']['xy'], 
                                    keypts_dict['left_shoulder']['conf_score'], keypts_dict['right_shoulder']['conf_score'], conf_threshold=conf_threshold,  part = "shoulders")
    hips = utils.get_mainpoint(keypts_dict['left_hip']['xy'], keypts_dict['right_hip']['xy'], keypts_dict['left_hip']['conf_score'], 
                                keypts_dict['right_hip']['conf_score'], conf_threshold=conf_threshold, part = "hips")
    knees = utils.get_mainpoint(keypts_dict['left_knee']['xy'], keypts_dict['right_knee']['xy'], keypts_dict['left_knee']['conf_score'], 
                                keypts_dict['right_knee']['conf_score'], conf_threshold=conf_threshold, part = "knees")
    
    ankles = utils.get_mainpoint(keypts_dict['left_ankle']['xy'], keypts_dict['right_ankle']['xy'], keypts_dict['left_ankle']['conf_score'], 
                                keypts_dict['right_ankle']['conf_score'], conf_threshold=conf_threshold, part = "ankles")    
    
    # if relevant keypt exist draw pt
    if shoulder:
        utils.draw_keypoint(frame, shoulder)
        # print(f'Shoulder: {shoulder}')
    if hips:
        utils.draw_keypoint(frame, hips)
        # print(f'Hips: {hips}')
    if knees:
        utils.draw_keypoint(frame, knees)
        # print(f'Knees: {knees}')
    
    if ankles:
        utils.draw_keypoint(frame, ankles)
        # print(f'Ankles: {ankles}')
        
    # if keypts exist draw line to connect them, calculate vector
    spine_vector = None
    legs_vector = None
    ankle_vector = None
    spine_vector_length = None
    legs_vector_length = None
    spine_leg_ratio = None
    
    # spine vector
    if shoulder and hips:
        spine_vector = utils.calculate_vector(hips, shoulder)
        # utils.draw_keypoint_line(frame, shoulder, hips)
        utils.draw_vector(frame, hips, spine_vector)
        # print(f'Spine Vector: {spine_vector}')
        spine_vector_length = np.linalg.norm(spine_vector)
    
    # leg vector
    if hips and knees:
        legs_vector = utils.calculate_vector(hips, knees)
        # legs_vector = utils.calculate_vector(knees, hips)
        # utils.draw_keypoint_line(frame, hips, knees)
        utils.draw_vector(frame, hips, legs_vector)
        # print(f'Leg Vector: {legs_vector}')
        legs_vector_length = np.linalg.norm(legs_vector)
    
    # ankle vector
    if knees and ankles:
        ankle_vector = utils.calculate_vector(knees, ankles)
        utils.draw_vector(frame, knees, ankle_vector)
    
    # spine-leg ratio
    if spine_vector_length is not None and legs_vector_length is not None:
        spine_leg_ratio = spine_vector_length/legs_vector_length
    
    
    # calculate vector if main pts exist
    spine_leg_theta = None # angle between spine (vector between shoulder and hips) and legs (vector between hips and knees)
    spine_x_axis_phi = None # angle between spine (vector between shoulder and hips) and x_axis along hip point
    legs_y_axis_alpha = None # angle between legs (vector between hips and knees) and y_axis along hip point
    ankle_beta = None # angle between ankle and x_axis along knee point
    if spine_vector and legs_vector:
        spine_leg_theta = utils.angle_between(spine_vector, legs_vector)
        hips_x_axis = utils.calculate_vector(hips, (hips[0]+20, hips[1]))
        hips_y_axis = utils.calculate_vector(hips, (hips[0], hips[1]+20))
        spine_x_axis_phi = utils.angle_between(spine_vector, hips_x_axis)
        legs_y_axis_alpha = utils.angle_between(legs_vector, hips_y_axis)
        if track_id != -1 and track_id != None:
            id_text =  "ID:" + str(track_id)
            cv2.putText(frame, id_text, (hips[0]+30, hips[1]-20),  cv2.FONT_HERSHEY_PLAIN,2,(155,200,0),2)
            if debug:
                print(f'Person ID: {track_id}')
    
    if legs_vector and ankle_vector:
        knee_x_axis = utils.calculate_vector(knees, (knees[0]+20, knees[1]))
        ankle_beta = utils.angle_between(ankle_vector, knee_x_axis)
        
    if spine_leg_theta and spine_x_axis_phi and legs_y_axis_alpha and ankle_beta and spine_leg_ratio and debug:
        print(f'Theta {spine_leg_theta}, Phi: {spine_x_axis_phi}, Alpha: {legs_y_axis_alpha}, Beta: {ankle_beta}, Spine-Leg Ratio: {spine_leg_ratio}')
        
    state = None
    # if at least have phi, alpha, and ratio, then can determine state
    if spine_x_axis_phi and legs_y_axis_alpha and spine_leg_ratio:
        state = utils.determine_state(theta=spine_leg_theta, phi=spine_x_axis_phi, alpha=legs_y_axis_alpha, beta=ankle_beta ,ratio=spine_leg_ratio)
        if debug:
            print(f'State: {state}')
        cv2.putText(frame, state, (hips[0]+30, hips[1]+20),  cv2.FONT_HERSHEY_PLAIN,2,(155,200,0),2)
    
    # return these for storing fall detection data
    return spine_vector, legs_vector, hips, shoulder, state

def extract_result(results, prev_data, curr_time, frame, debug=False):
    """ 
    Go through Yolo pose results, calls extract_keypts to get keypoints and 
    relevant vector/angles and state, then appends relevant information to prev_data dictionary.
    
    Inputs:
    - results: list of YOLO results
    - prev_data: pointer to dictionary of previous frame data
    - curr_time: current time
    - frame: frame to draw on
    
    """
    # get data from inference
    for result in results:
        keypts = result.keypoints
        # print(f'Keypoints: \n{kpts}')
        num_people = keypts.shape[0]
        num_pts = keypts.shape[1]
        track_id = [1, 1, 1] # random pad for testing
        boxes = None
        if result.boxes.id is not None:
            track_id = result.boxes.id.tolist()
            boxes = result.boxes.xywh.tolist()            
        
        # if keypts detected
        if num_pts !=0:
            # for each person
            for i in range(num_people):
                id = int(track_id[i])
                spine_vector, leg_vector, hips, shoulders, state = extract_keypts(i, keypts, frame, conf_threshold=0.5,track_id=id, debug=debug)
                if spine_vector:
                    # append spine vector to prev_data
                    if prev_data.get(id) == None:
                        prev_data[id] = {}
                        prev_data[id]['spine_vector'] = [spine_vector]
                        prev_data[id]['hips'] = [hips]
                        prev_data[id]['shoulders'] = [shoulders]
                        prev_data[id]['state'] = [state]
                    else:
                        if len(prev_data[id]['spine_vector'])>=3:
                            prev_data[id]['spine_vector'].pop(0)
                            prev_data[id]['hips'].pop(0)
                            prev_data[id]['shoulders'].pop(0)
                        prev_data[id]['spine_vector'].append(spine_vector)
                        prev_data[id]['hips'].append(hips)
                        prev_data[id]['shoulders'].append(shoulders)
                        prev_data[id]['state'].append(state)
                    # append inference time
                    prev_data[id]['last_check'] = curr_time
    

def fall_detection(prev_data, curr_time,frame, debug=False):
    """ 
    Fall detection algorithm using previous 3 frame data.
    
    Input:
    - prev_data: dictionary of previous frame data
    - curr_time: current time
    - frame: frame to draw on
    
    """
    # fall detection using previous frames
    for key, value in prev_data.copy().items():
        # detect fall from spine vector in past 3 frames
        if len(value['spine_vector']) == 3:
            fall_angle1 = utils.angle_between(value['spine_vector'][0], value['spine_vector'][1])
            fall_angle2 = utils.angle_between(value['spine_vector'][1], value['spine_vector'][2])
            fall_angle3 = utils.angle_between(value['spine_vector'][0], value['spine_vector'][2])
            hip_diff1 = abs(prev_data[key]["hips"][0][1] - prev_data[key]["hips"][1][1])
            hip_diff2 = abs(prev_data[key]["hips"][1][1] - prev_data[key]["hips"][2][1])
            hip_diff3 = abs(prev_data[key]["hips"][0][1] - prev_data[key]["hips"][2][1])
            shoulder_diff1 = prev_data[key]["shoulders"][0][1] - prev_data[key]["shoulders"][1][1]
            shoulder_diff2 = prev_data[key]["shoulders"][1][1] - prev_data[key]["shoulders"][2][1]
            shoulder_diff3 = prev_data[key]["shoulders"][0][1] - prev_data[key]["shoulders"][2][1]
            # large angle somewhere between spine vector likely indicates fall
            if (fall_angle1 > 50 or fall_angle2 > 50 or fall_angle3 > 50) and (shoulder_diff1 <= 0 or shoulder_diff2 <=0 or shoulder_diff3 <= 0):
                if hip_diff1 >= 10 or hip_diff2 >=10 or hip_diff3 >= 10:
                    fall_detected = True
                    print("large hip diff")
                    print(f'shoulders: {prev_data[key]["shoulders"]}')
                    print(f'angles: {fall_angle1}, {fall_angle2}, {fall_angle3}')
                    print(f'spine: {prev_data[key]["spine_vector"]}')
                    print(f'hips: {prev_data[key]["hips"]}')
                    print(f"Person {key} Fall Detected")
                    cv2.putText(frame, "Fall Detected", (prev_data[key]["hips"][2][0]+30, prev_data[key]["hips"][2][1]+50),  cv2.FONT_HERSHEY_PLAIN,2,(245,0,0),2)
                    # remove data from prev_data to avoid multiple detections
                    prev_data[key]['spine_vector'] = prev_data[key]['spine_vector'][3:]
                    prev_data[key]['hips'] = prev_data[key]['hips'][3:]
                    prev_data[key]['shoulders'] = prev_data[key]['shoulders'][3:]
                else:
                    print("Low probability of fall")
                    print(f'angles: {fall_angle1}, {fall_angle2}, {fall_angle3}')
                    print(f'spine: {prev_data[key]["spine_vector"]}')
                    print(f'hips: {prev_data[key]["hips"]}')
        
        # delete data if not checked for a while
        time_till_delete = 2
        if curr_time -  value['last_check'] > time_till_delete:
            if debug:
                print(f"ID {key} hasn't checked over {time_till_delete} seconds")
                print(f'Deleting ID {key}')
            del prev_data[key]
            


def stream_inference(vid_source="/dev/video0", vid_width=640, vid_height=640, show_frame=True, manual_move=False, interval=0, debug=False):
    """ Runs inference with threading, for usb camera stream.  """
    print("Running inference with threading on usb camera.")    
    # load pretrained model
    model = YOLO("yolo-weights/yolov8n-pose.pt")

    # non threaded usb camera stream
    # cap = cv2.VideoCapture(vid_source)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, vid_width)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, vid_height)
    
    # threaded usb cam stream
    cap = USBCamStream(src=vid_source)
    cap.resize_stream(vid_width,vid_height) 
    cap = cap.start()
    # cap.change_format()

    # dictionary to store prev frame data for determining action
    prev_data = {}

    # time variables
    prev_time = 0 # for tracking interval to execute predict with Yolo model
    prev_time_fps = 0 # for fps counting
    num_frames_elapsed = 0
    
    while True:
        # read frame
        ret, frame = cap.read()
        if not ret:
            break
        
        # track time for interval and fps
        curr_time = time.time()
        # elapsed_time = curr_time - prev_time
        num_frames_elapsed += 1
        # if specified predict interval
        # if elapsed_time >= interval:
        if num_frames_elapsed >= interval:
            # update timer
            # prev_time = curr_time
            num_frames_elapsed = 0
            # inference
            # results = model.predict(frame, imgsz=640, conf=0.5, verbose=False)
            results = model.track(frame, imgsz=640, conf=0.6, verbose=False, tracker="bytetrack.yaml", persist=True)
            extract_result(results, prev_data, curr_time, frame, debug=debug)
            fall_detection(prev_data, curr_time, frame, debug=debug)
            
        
        
        # track fps and draw to frame
        # fps = 1/(curr_time-prev_time_fps)
        # cv2.putText(frame, str(int(fps)), (50,50),  cv2.FONT_HERSHEY_PLAIN,3,(225,0,0),3)
        # prev_time_fps = curr_time
        
        # show frame to screen
        if show_frame:
            cv2.imshow('Yolo Pose Test', frame)
        
        # wait for user key
        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break
        
    # cleanup
    cap.stop()
    cap.release()
    cv2.destroyAllWindows()


def video_inference(vid_source="./test-data/videos/fall-1.mp4", vid_width=640, vid_height=640, show_frame=True, manual_move=False, interval=0, debug=False):
    """ Runs inference on video without threading """    
    # load pretrained model
    model = YOLO("yolo-weights/yolov8n-pose.pt")

    # non threaded usb camera stream
    cap = cv2.VideoCapture(vid_source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, vid_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, vid_height)
    
    # threaded usb cam stream

    # dictionary to store prev frame data for determining action
    prev_data = {}

    # time variables
    prev_time = 0 # for tracking interval to execute predict with Yolo model
    prev_time_fps = 0 # for fps counting
    num_frames_elapsed = 0
    
    while True:
        # read frame
        ret, frame = cap.read()
        if not ret:
            break
        
        # track time for interval and fps
        curr_time = time.time()
        # elapsed_time = curr_time - prev_time
        num_frames_elapsed += 1
        fall_detected = False
        fall_draw_points = []
        
        # if specified predict interval
        # if elapsed_time >= interval:
        if num_frames_elapsed >= interval:
            # update timer
            # prev_time = curr_time
            num_frames_elapsed = 0
            
            # inference
            # results = model.predict(frame, imgsz=640, conf=0.5, verbose=False)
            results = model.track(frame, imgsz=640, conf=0.6, verbose=False, tracker="bytetrack.yaml", persist=True)

            # Visualize the results on the frame
            # frame = results[0].plot()
            
            extract_result(results, prev_data, curr_time, frame, debug=debug)
            fall_detection(prev_data, curr_time, frame, debug=debug)
            
                        
        # track fps and draw to frame
        fps = 1/(curr_time-prev_time_fps)
        cv2.putText(frame, str(int(fps)), (50,50),  cv2.FONT_HERSHEY_PLAIN,3,(225,0,0),3)
        prev_time_fps = curr_time
        
        # show frame to screen
        if show_frame:
            cv2.imshow('Yolo Pose Test', frame)
        
        # wait for user key
        key = cv2.waitKey(10)
        if manual_move:
            key = cv2.waitKey(0)
        # press esc to quit
        if key == 27:  # ESC
            break
        
    # cleanup
    cap.release()
    cv2.destroyAllWindows()


def image_inference(img_src="/dev/video0", width=640, height=640, show_frame=True, debug=False):
    model = YOLO("yolo-weights/yolov8n-pose.pt")
    frame = cv2.imread(img_src)
    frame = cv2.resize(frame, (640,640))
    results = model.track(frame, conf=0.5, persist=True, tracker="bytetrack.yaml")
    
    # Visualize the results on the frame
    # frame = results[0].plot()
    for result in results:
        keypts = result.keypoints
        # print(f'Keypoints: \n{kpts}')
        num_people = keypts.shape[0]
        num_pts = keypts.shape[1]
        track_id = [1, 1, 1]
        if result.boxes.id is not None:
            track_id = result.boxes.id.tolist()
            boxes = result.boxes.xywh.tolist()   
        
        if num_pts!=0:
            for i in range(num_people):
                extract_keypts(i, keypts, frame, conf_threshold=0.5, track_id=int(track_id[i]), debug=debug)
    
    cv2.imshow('Yolo Pose Test', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()