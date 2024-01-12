# This file contains some basic math, drawing, and other utils to share with other files

import argparse
import math
import numpy as np
import cv2

def get_args():
    """ Parse arguments. Returns user/default set arguments """
    parser = argparse.ArgumentParser(description="Pose estimate program")
    # user settings
    parser.add_argument("--src", type=str, default='./test-data/videos/fall-1.mp4', help="Video file location (eg. ./test-data/videos/fall-1.mp4)")
    parser.add_argument("--show", type=int, default=1, help="Whether to show camera to screen, 0 to hide 1 to show.")
    parser.add_argument("--width", type=int, default=480, help="Input video width. (eg. 480)")
    parser.add_argument("--height", type=int, default=480, help="Input video height (eg. 480)")
    parser.add_argument("--conf_score", type=float, default=0.5, help="Confidence score threshold (eg. 0.7)")
    parser.add_argument("--interval", type=float, default=0, help="Interval in seconds to run inference (eg. 2)")
    parser.add_argument("--manual_frame", type=float, default=0, help="Set this to 1 if you want to press 'n' key to advance each video frame.")
    parser.add_argument("--type", type=int, default=0, help="Specifies whether input is image or video (0 for video 1 for image). Default is video (0).")
    parser.add_argument("--debug", type=int, default=0, help="Whether to print some debug info. Default is 0 (no debug info), 1 means print debug info.")
    parser.add_argument("--save_vid", type=int, default=0, help="Whether to save video. Default is 0 (no save), 1 means save video.")
    
    args = parser.parse_args()
    return args

def calculate_midpoint(point1, point2):
    """ Calculate midpoint between two 2D points. Returns midpoint (x,y) """
    x1, y1 = point1
    x2, y2 = point2
    midpoint_x = int((x1 + x2) / 2)
    midpoint_y = int((y1 + y2) / 2)
    return (midpoint_x, midpoint_y)

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle (deg) between vectors v1 and v2 """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return math.degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))

def get_mainpoint(left, right, left_conf_score, right_conf_score, conf_threshold=0.5, part="part"):
    """ 
    For each important part (eg. hip), if both left and right part exists and 
    confidence score is past threshold, we get midpoint. 
    
    If only left or right exist, then we set the point to left or right. 
    
    Returns midpoint (x,y) coordinate
    """
    main_point = None
    if left != (0,0) and right != (0,0) and left_conf_score>conf_threshold and right_conf_score>conf_threshold:
        # print(f'both left and right {part} detected')
        main_point = calculate_midpoint(left, right)
    elif left != (0,0) and left_conf_score>conf_threshold:
        # print(f'only left {part} detected')
        main_point = left
    elif right != (0,0) and right_conf_score>conf_threshold:
        # print(f'only right {part} detected')
        main_point = right
    return main_point


def calculate_vector(point1, point2):
    """ Returns 2D vector between two points (x,y). """
    x1, y1 = point1
    x2, y2 = point2
    vector_x = x2 - x1
    vector_y = y2 - y1
    return (vector_x, vector_y)

def calculate_distance(point1, point2):
    """ Returns distance between two points as float. """
    x1, y1 = point1
    x2, y2 = point2
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance

def draw_keypoint(image, keypoint, color=(0, 255, 0), thickness=-1):
    """ Takes image and keypoint to use OpenCV to draw pose keypoints on image and returns annotated image. """
    cv2.circle(image, (keypoint[0], keypoint[1]), 5, color, thickness)

def draw_keypoint_line(image, keypoint1, keypoint2):
    """ Draws line between two keypoints onto image and returns annotated image. """
    cv2.line(image, keypoint1, keypoint2, (255, 255, 255), 4)

def draw_vector(image, start_point, vector, color=(255, 255, 255), thickness=5):
    """ Draws vector to image with OpenCV line. Takes in starting point, vector, and image to draw on.   """
    end_point = tuple(np.array(start_point) + np.array(vector))
    cv2.line(image, tuple(start_point), end_point, color, thickness)


def determine_state(phi=None, alpha=None, ratio=None, beta=None, theta=None):
    """ 
    Test function for checking action state. Still trying out. 
    
    Input:
    - theta: angle between spine and legs
    - alpha: angle between legs and y-axis along hips
    - phi: angle between spine and x-axis along hips
    - beta: angle between legs and ankle
    - ratio: ratio between legs and spine vector lengths
    """
    # if no legs
    if beta==None:
        # legs super close to hips usually means person is sitting (eg. cross legged, sitting directly in front of camera, etc.)
        if ratio!=None and ratio>2:
            return "sitting"
        # alpha<=30 means legs are vertical, usually means person is standing/walking
        if alpha != None and alpha<=27:
            return "standing"
        # (phi<=25 or phi>=155) means spine is parallel to ground, usually means person is lying down
        if phi != None and (phi<=30 or phi>=150):
            return "lying down"
        # otherwise most likely sitting
        else:
            return "sitting"
            # raise Exception("invalid theta angle")
    else:
        # legs super close to hips usually means person is sitting (eg. cross legged, sitting directly in front of camera, etc.)
        if ratio!=None and ratio>2:
            if phi != None and (phi<=35 or phi>=145):
                return "lying down"
            return "sitting"
        # alpha<=30 means legs are vertical, usually means person is standing/walking
        if alpha != None and alpha<=27 and beta>=65 and beta<=125:
            return "standing"
        # (phi<=25 or phi>=155) means spine is parallel to ground, usually means person is lying down
        if phi != None and (phi<=35 or phi>=145):
            return "lying down"
        # otherwise most likely sitting
        if theta<165:
            return "sitting"
        else:
            return "lying down"
            # raise Exception("invalid theta angle")
            
def fall_detection(prev_data, curr_time,frame=None, debug=False):
    """ 
    Fall detection algorithm using previous 3 frame data.
    
    Input:
    - prev_data: dictionary of previous frame data
    - curr_time: current time
    - frame: frame to draw on (optional)
    
    Output:
    - fall_detected: boolean indicating whether fall detected
    - fall_conf: confidence score of fall detection (still tweaking)
    
    """
    fall_detected = False
    fall_conf = 0
    angle_threshold = 50
    shoulder_threshold = 0
    hip_threshold = 22
    # fall detection using previous frames
    for key, value in prev_data.copy().items():
        # detect fall from spine vector in past 3 frames
        if len(value['spine_vector']) == 3:
            # calculate angles between spine vectors
            fall_angle1 = angle_between(value['spine_vector'][0], value['spine_vector'][1])
            fall_angle2 = angle_between(value['spine_vector'][1], value['spine_vector'][2])
            fall_angle3 = angle_between(value['spine_vector'][0], value['spine_vector'][2])
            max_angle = max(fall_angle1, fall_angle2, fall_angle3)
            # calculate difference between hips, large difference likely indicates fall
            hip_diff1 = abs(prev_data[key]["hips"][0][1] - prev_data[key]["hips"][1][1])
            hip_diff2 = abs(prev_data[key]["hips"][1][1] - prev_data[key]["hips"][2][1])
            hip_diff3 = abs(prev_data[key]["hips"][0][1] - prev_data[key]["hips"][2][1])
            max_hip_diff = max(hip_diff1, hip_diff2, hip_diff3)
            # calculate difference between shoulders, large difference likely indicates fall
            shoulder_diff1 = prev_data[key]["shoulders"][0][1] - prev_data[key]["shoulders"][1][1]
            shoulder_diff2 = prev_data[key]["shoulders"][1][1] - prev_data[key]["shoulders"][2][1]
            shoulder_diff3 = prev_data[key]["shoulders"][0][1] - prev_data[key]["shoulders"][2][1]
            min_shoulder_diff = min(shoulder_diff1, shoulder_diff2, shoulder_diff3)
            
            state = prev_data[key]['state'][-1]
            
            
            # large angle somewhere between spine vector likely indicates fall
            if ((max_angle>angle_threshold and min_shoulder_diff<=shoulder_threshold) or \
                (min_shoulder_diff<-100 and max_angle>30) or \
                    (max_angle>23 and min_shoulder_diff<-30 and max_hip_diff>50)) and \
                        state!="standing":
                # if hip changed a lot, likely indicates fall
                if max_hip_diff>hip_threshold:
                    fall_detected = True
                    # random confidence, need to adjust
                    fall_conf = min(1, max(0.7, 0.4*max_angle/80 + 0.4*max_hip_diff/50))
                    print(f"\nHigh Probability of Person {key} Fall Detected!!")
                                        
                    # debug stuff
                    if debug:
                        print(f'Confidence: {fall_conf}')
                        print(f'State: {state}')
                        print(f"Hip Diffs: {max_hip_diff}, Shoulder diffs: {min_shoulder_diff}")
                        print(f'angles: {fall_angle1}, {fall_angle2}, {fall_angle3}')
                        print(f'spine: {prev_data[key]["spine_vector"]}')
                        print(f'hips: {prev_data[key]["hips"]}')
                        print(f'shoulders: {prev_data[key]["shoulders"]}')
                        print()
                        # cv2.putText(frame, "Fall Detected", (prev_data[key]["hips"][2][0]+30, prev_data[key]["hips"][2][1]+50),  cv2.FONT_HERSHEY_PLAIN,2,(245,0,0),2)
                    
                    # remove data from prev_data to avoid multiple detections
                    prev_data[key]['spine_vector'] = prev_data[key]['spine_vector'][3:]
                    prev_data[key]['hips'] = prev_data[key]['hips'][3:]
                    prev_data[key]['shoulders'] = prev_data[key]['shoulders'][3:]
                    return fall_detected, fall_conf
                # otherwise spine changed a lot but likely no fall
                else:
                    fall_detected = True
                    # random confidence, need to adjust
                    fall_conf = min(0.4, 0.35*max_angle/80 + 0.35*max_hip_diff/50)
                    print(f"Low Probability of Person {key} Fall Detected.")
                    if debug:
                        print(f'Confidence: {fall_conf}')
                        print(f'State: {state}')
                        print(f"Hip Diffs: {max_hip_diff}, Shoulder diffs: {min_shoulder_diff}")
                        print(f'angles: {fall_angle1}, {fall_angle2}, {fall_angle3}')
                        print(f'spine: {prev_data[key]["spine_vector"]}')
                        print(f'hips: {prev_data[key]["hips"]}')
                        print()
                    return fall_detected, fall_conf
            
        # delete data if not checked for a while
        time_till_delete = 2
        if curr_time -  value['last_check'] > time_till_delete:
            if debug:
                print(f"ID {key} hasn't checked over {time_till_delete} seconds")
                print(f'Deleting ID {key}')
            del prev_data[key]
    
    return fall_detected, fall_conf



def fall_detection_v2(prev_data, curr_time,frame=None, debug=False):
    """ 
    Fall detection algorithm using previous 3 frame data.
    
    Input:
    - prev_data: dictionary of previous frame data
    - curr_time: current time
    - frame: frame to draw on (optional)
    
    Output:
    - fall_detected: boolean indicating whether fall detected
    - fall_conf: confidence score of fall detection (still tweaking)
    
    """
    fall_detected = False
    fall_conf = 0
    angle_threshold = 50
    shoulder_threshold = 0
    hip_threshold = 22
    # fall detection using previous frames
    for key, value in prev_data.copy().items():
        # detect fall from spine vector in past 3 frames
        if len(value['spine_vector']) == 3:
            # calculate angles between spine vectors
            fall_angle1 = angle_between(value['spine_vector'][0], value['spine_vector'][1])
            fall_angle2 = angle_between(value['spine_vector'][1], value['spine_vector'][2])
            fall_angle3 = angle_between(value['spine_vector'][0], value['spine_vector'][2])
            max_angle = max(fall_angle1, fall_angle2, fall_angle3)
            # calculate difference between hips, large difference likely indicates fall
            hip_diff1 = abs(prev_data[key]["hips"][0][1] - prev_data[key]["hips"][1][1])
            hip_diff2 = abs(prev_data[key]["hips"][1][1] - prev_data[key]["hips"][2][1])
            hip_diff3 = abs(prev_data[key]["hips"][0][1] - prev_data[key]["hips"][2][1])
            max_hip_diff = max(hip_diff1, hip_diff2, hip_diff3)
            # calculate difference between shoulders, large difference likely indicates fall
            shoulder_diff1 = prev_data[key]["shoulders"][0][1] - prev_data[key]["shoulders"][1][1]
            shoulder_diff2 = prev_data[key]["shoulders"][1][1] - prev_data[key]["shoulders"][2][1]
            shoulder_diff3 = prev_data[key]["shoulders"][0][1] - prev_data[key]["shoulders"][2][1]
            min_shoulder_diff = min(shoulder_diff1, shoulder_diff2, shoulder_diff3)
            
            state = prev_data[key]['state'][-1]
            if state == "standing":
                state_conf = -2
            else:
                state_conf = 2
            
            fall_conf = max_angle/angle_threshold + max_hip_diff/30 + min_shoulder_diff/-40 + state_conf
            
            # print(f'Fall Conf: {fall_conf}')
            
            if fall_conf>8:
                fall_detected = True
                print(f"\nHigh Probability of Person {key} Fall Detected!!")
                # remove data from prev_data to avoid multiple detections
                prev_data[key]['spine_vector'] = prev_data[key]['spine_vector'][3:]
                prev_data[key]['hips'] = prev_data[key]['hips'][3:]
                prev_data[key]['shoulders'] = prev_data[key]['shoulders'][3:]
                if debug:
                    print(f'Confidence: {fall_conf}')
                    print(f'State: {state}')
                    print(f"Hip Diffs: {max_hip_diff}, Shoulder diffs: {min_shoulder_diff}")
                    print(f'angles: {fall_angle1}, {fall_angle2}, {fall_angle3}')
                    print(f'spine: {prev_data[key]["spine_vector"]}')
                    print(f'hips: {prev_data[key]["hips"]}')
                    print()
            elif fall_conf<=8 and fall_conf>5:
                fall_detected = True
                print(f"Low Probability of Person {key} Fall Detected.")
                if debug:
                    print(f'Confidence: {fall_conf}')
                    print(f'State: {state}')
                    print(f"Hip Diffs: {max_hip_diff}, Shoulder diffs: {min_shoulder_diff}")
                    print(f'angles: {fall_angle1}, {fall_angle2}, {fall_angle3}')
                    print(f'spine: {prev_data[key]["spine_vector"]}')
                    print(f'hips: {prev_data[key]["hips"]}')
                    print()
            
            fall_conf = min(1,fall_conf/15)
            
        # delete data if not checked for a while
        time_till_delete = 2
        if curr_time -  value['last_check'] > time_till_delete:
            if debug:
                print(f"ID {key} hasn't checked over {time_till_delete} seconds")
                print(f'Deleting ID {key}')
            del prev_data[key]
    
    return fall_detected, fall_conf