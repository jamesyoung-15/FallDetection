# import the necessary packages
from ultralytics import YOLO
import cv2
import time
import utils
import my_defs
import numpy as np
from camera_stream.usbcam_stream import USBCamStream
from collections import defaultdict


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
    

def inference(person_num, keypts, frame, conf_threshold=0.5, track_id=0):
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
    
    # track if main parts exist
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
    spine_vector_length = None
    legs_vector_length = None
    spine_leg_ratio = None
    if shoulder_exist and hips_exist:
        spine_vector = utils.calculate_vector(hips, shoulder)
        # utils.draw_keypoint_line(frame, shoulder, hips)
        utils.draw_vector(frame, hips, spine_vector)
        # print(f'Spine Vector: {spine_vector}')
        spine_vector_length = np.linalg.norm(spine_vector)
        
    if hips_exist and knees_exist:
        legs_vector = utils.calculate_vector(hips, knees)
        # legs_vector = utils.calculate_vector(knees, hips)
        # utils.draw_keypoint_line(frame, hips, knees)
        utils.draw_vector(frame, hips, legs_vector)
        # print(f'Leg Vector: {legs_vector}')
        legs_vector_length = np.linalg.norm(legs_vector)
    
    if spine_vector_length is not None and legs_vector_length is not None:
        spine_leg_ratio = spine_vector_length/legs_vector_length
    
    # print(f'Spine Vector Length: {spine_vector_length}, Legs Vector Length: {legs_vector_length}, Ratio: {spine_leg_ratio}')
    
    # calculate vector if all 3 main pts exist
    spine_leg_theta = None # angle between spine (vector between shoulder and hips) and legs (vector between hips and knees)
    spine_x_axis_phi = None # angle between spine (vector between shoulder and hips) and x_axis along hip point
    legs_y_axis_alpha = None # angle between legs (vector between hips and knees) and y_axis along hip point
    if shoulder_exist and hips_exist and knees_exist:
        spine_leg_theta = utils.angle_between(spine_vector, legs_vector)
        hips_x_axis = utils.calculate_vector(hips, (hips[0]+20, hips[1]))
        hips_y_axis = utils.calculate_vector(hips, (hips[0], hips[1]+20))
        # utils.draw_vector(frame, hips, hips_x_axis, color=(255,255,255))
        # spine_x_axis_phi = utils.calculate_angle_with_x_axis(spine_vector)
        spine_x_axis_phi = utils.angle_between(spine_vector, hips_x_axis)
        legs_y_axis_alpha = utils.angle_between(legs_vector, hips_y_axis)
        print(f'Person ID: {track_id}')
        print(f'Theta {spine_leg_theta}, Phi: {spine_x_axis_phi}, Alpha: {legs_y_axis_alpha}')
        # state = utils.action_state(spine_leg_theta, spine_x_axis_phi, legs_y_axis_alpha)
        state = utils.test_state(theta=spine_leg_theta, phi=spine_x_axis_phi, alpha=legs_y_axis_alpha, ratio=spine_leg_ratio)
        print(f'State: {state}')
        if track_id != -1:
            id_text =  "ID:" + str(track_id)
            cv2.putText(frame, id_text, (hips[0]+30, hips[1]-20),  cv2.FONT_HERSHEY_PLAIN,2,(155,200,0),2)
        cv2.putText(frame, state, (hips[0]+30, hips[1]+20),  cv2.FONT_HERSHEY_PLAIN,2,(155,200,0),2)
        
    # test fall detect
    return spine_vector, hips

def stream_inference(vid_source="/dev/video0", vid_width=640, vid_height=640, show_frame=True, manual_move=False, interval=0):
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

    # array to store prev frame data for determining action
    # prev_data = [None]

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
            results = model.track(frame, imgsz=640, conf=0.5, verbose=False, tracker="bytetrack.yaml", persist=True)
            # get data from inference
            for result in results:
                keypts = result.keypoints
                # print(f'Keypoints: \n{kpts}')
                num_people = keypts.shape[0]
                num_pts = keypts.shape[1]
                track_id = [-1, -1, -1] # random pad for testing
                boxes = None
                if result.boxes.id is not None:
                    track_id = result.boxes.id.tolist()
                    boxes = result.boxes.xywh.tolist()            
                
                # if keypts detected
                if num_pts !=0:
                    for i in range(num_people):
                        inference(i, keypts, frame, conf_threshold=0.5, track_id=int(track_id[i]))

                            
        # track fps and draw to frame
        fps = 1/(curr_time-prev_time_fps)
        cv2.putText(frame, str(int(fps)), (50,50),  cv2.FONT_HERSHEY_PLAIN,3,(225,0,0),3)
        prev_time_fps = curr_time
        
        # show frame to screen
        if show_frame:
            cv2.imshow('Yolo Pose Test', frame)
        
        # wait for user key
        key = cv2.waitKey(1)
        # if manual move, press n to move to next frame
        # if manual_move:
        #     key = cv2.waitKey(0)
        #     if key == ord("n"):
        #         continue
        # press esc to quit
        if key == 27:  # ESC
            break
        
    # cleanup
    cap.stop()
    cap.release()
    cv2.destroyAllWindows()


def video_inference(vid_source="./test-data/videos/fall-1.mp4", vid_width=640, vid_height=640, show_frame=True, manual_move=False, interval=0):
    """ Runs inference on video without threading """    
    # load pretrained model
    model = YOLO("yolo-weights/yolov8n-pose.pt")

    # non threaded usb camera stream
    cap = cv2.VideoCapture(vid_source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, vid_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, vid_height)
    
    # threaded usb cam stream

    # array to store prev frame data for determining action
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
            results = model.track(frame, imgsz=640, conf=0.5, verbose=False, tracker="bytetrack.yaml", persist=True)

            # Visualize the results on the frame
            # frame = results[0].plot()
            
            
            # get data from inference
            for result in results:
                keypts = result.keypoints
                # print(f'Keypoints: \n{kpts}')
                num_people = keypts.shape[0]
                num_pts = keypts.shape[1]
                
                track_id = [-1, -1, -1] # random pad for testing
                boxes = None
                
                if result.boxes.id is not None:
                    track_id = result.boxes.id.tolist()
                    boxes = result.boxes.xywh.tolist()
                
                # if keypts detected
                if num_pts !=0:
                    # for each person
                    for i in range(num_people):
                        id = int(track_id[i])
                        spine_vector, hips = inference(i, keypts, frame, conf_threshold=0.5,track_id=id)
                        # append spine vector to prev_data
                        if prev_data.get(id) == None:
                            prev_data[id] = {}
                            prev_data[id]['spine_vector'] = [spine_vector]
                        else:
                            if len(prev_data[id]['spine_vector'])>=3:
                                prev_data[id]['spine_vector'].pop(0)
                            prev_data[id]['spine_vector'].append(spine_vector)
                        # append inference time
                        prev_data[id]['last_check'] = curr_time
            
            for key, value in prev_data.copy().items():
                # detect fall from spine vector in past 3 frames
                if len(value['spine_vector']) == 3:
                    # large angle somewhere between spine vector likely indicates fall
                    if utils.angle_between(value['spine_vector'][0], value['spine_vector'][1]) > 50 \
                    or utils.angle_between(value['spine_vector'][1], value['spine_vector'][2]) > 50 \
                    or utils.angle_between(value['spine_vector'][0], value['spine_vector'][2]) > 50:
                        print(prev_data)
                        print(f"Person {key} Fall Detected")
                        cv2.putText(frame, "Fall Detected", (hips[0]+30, hips[1]+50),  cv2.FONT_HERSHEY_PLAIN,2,(245,0,0),2)
                
                # delete data if not checked for a while
                time_till_delete = 2
                if curr_time -  value['last_check'] > time_till_delete:
                    print(f"ID {key} hasn't checked over {time_till_delete} seconds")
                    print(f'Deleting ID {key}')
                    del prev_data[key]
                        
                        
        # track fps and draw to frame
        fps = 1/(curr_time-prev_time_fps)
        cv2.putText(frame, str(int(fps)), (50,50),  cv2.FONT_HERSHEY_PLAIN,3,(225,0,0),3)
        prev_time_fps = curr_time
        
        # show frame to screen
        if show_frame:
            cv2.imshow('Yolo Pose Test', frame)
        
        # wait for user key
        key = cv2.waitKey(10)
        # if manual move, press n to move to next frame
        if manual_move:
            key = cv2.waitKey(0)
            if key == ord("n"):
                continue
        # press esc to quit
        if key == 27:  # ESC
            break
        
    # cleanup
    cap.release()
    cv2.destroyAllWindows()


def image_inference(img_src="/dev/video0", width=640, height=640, show_frame=True):
    model = YOLO("yolo-weights/yolov8n-pose.pt")
    frame = cv2.imread(img_src)
    frame = cv2.resize(frame, (640,640))
    results = model.predict(frame, conf=0.5)
    for result in results:
        keypts = result.keypoints
        # print(f'Keypoints: \n{kpts}')
        num_people = keypts.shape[0]
        num_pts = keypts.shape[1]
        if num_pts!=0:
            for i in range(num_people):
                inference(i, keypts, frame, conf_threshold=0.5)
    
    cv2.imshow('Yolo Pose Test', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()