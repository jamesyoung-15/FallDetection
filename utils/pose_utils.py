# This file contains some basic math, drawing, and other utils to share with other files

import argparse
import math
import numpy as np
import cv2

def get_args():
    """ Parse arguments. Returns user/default set arguments """
    parser = argparse.ArgumentParser(description="Pose estimate program")
    # user settings
    parser.add_argument("--src", type=str, default='./test-data/videos/fall-1.mp4', help="Video file location (eg. /dev/video0)")
    parser.add_argument("--show", type=int, default=1, help="Whether to show camera to screen, 0 to hide 1 to show.")
    parser.add_argument("--width", type=int, default=640, help="Input video width. (eg. 640)")
    parser.add_argument("--height", type=int, default=480, help="Input video height (eg. 480)")
    parser.add_argument("--conf_score", type=float, default=0.5, help="Confidence score threshold (eg. 0.7)")
    parser.add_argument("--interval", type=int, default=5, help="Interval in frames to run inference (eg. 2 means inference every 2 frames)")
    parser.add_argument("--manual_frame", type=int, default=0, help="Set this to 1 if you want to press 'n' key to advance each video frame.")
    parser.add_argument("--type", type=int, default=0, help="Specifies whether input is image or video (0 for video 1 for image). Default is video (0).")
    parser.add_argument("--debug", type=int, default=0, help="Whether to print some debug info. Default is 0 (no debug info), 1 means print debug info.")
    parser.add_argument("--save_vid", type=int, default=0, help="Whether to save video. Default is 0 (no save), 1 means save video.")
    parser.add_argument("--pose_type", type=int, default=0, help="Specify which pose model to use. 0 for YoloV8Pose (default), \
                                                                    1 for Movenet Multi Lightning.")
    parser.add_argument("--resize_frame", type=int, default=0, help="Whether to resize frame. Default is 0 (no resize), 1 means resize frame.")
    parser.add_argument("--delay", type=int, default=1, help="Delay in ms for cv2.waitkey(delay) in int.")
    parser.add_argument("--fps", type=int, default=24, help="Set FPS for cv2 (only for usb camera).")
    parser.add_argument("--benchmark", type=int, default=0, help="Record and print FPS after exit. 0 for false 1 for true.")
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


def determine_state(phi=None, alpha=None, ratio=None, beta=None, theta=None, shoulder=None, knees=None):
    """ 
    Test function for checking action state. Still trying out. 
    
    Args:
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
        # shoulder below knees usually means person is lying down
        if (shoulder and knees) and (shoulder[1]>knees[1]):
            return "lying down"
        # legs super close to hips usually means person is sitting (eg. cross legged, sitting directly in front of camera, etc.)
        elif ratio!=None and ratio>2:
            if phi != None and (phi<=35 or phi>=145):
                return "lying down"
            return "sitting"
        # alpha<=30 means legs are vertical, usually means person is standing/walking
        elif (alpha != None and alpha<31) and (beta>=65 and beta<=125):
            return "standing"
        # (phi<=25 or phi>=155) means spine is parallel to ground, usually means person is lying down
        elif phi != None and (phi<=35 or phi>=145):
            return "lying down"
        # otherwise most likely sitting
        elif theta<160:
            return "sitting"
        else:
            return "lying down"
            # raise Exception("err")