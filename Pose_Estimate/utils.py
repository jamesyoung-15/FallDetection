# This file contains some basic math, drawing, and other utils to share with other files

import argparse
import math
import numpy as np
import cv2

def get_args():
    """ Parse arguments. Returns user/default set arguments """
    parser = argparse.ArgumentParser(description="Pose estimate program")
    # user settings
    parser.add_argument("--src", type=str, default='/dev/video0', help="Video file location (eg. /dev/video0)")
    parser.add_argument("--show", type=int, default=1, help="Whether to show camera to screen, 0 to hide 1 to show.")
    parser.add_argument("--width", type=int, default=640, help="Input video width.")
    parser.add_argument("--height", type=int, default=480, help="Input video height")
    parser.add_argument("--conf_score", type=float, default=0.4)
    parser.add_argument("--interval", type=float, default=0, help="Interval in seconds to run inference (eg. 2)")
    parser.add_argument("--type", type=int, default=0, help="Specifies whether input is image or video (0 for video 1 for image). Default is video (0).")
    
    
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

def draw_keypoint(image, keypoint):
    """ Takes image and keypoint to use OpenCV to draw pose keypoints on image and returns annotated image. """
    cv2.circle(image, (keypoint[0], keypoint[1]), 5, (0, 255, 0), -1)

def draw_keypoint_line(image, keypoint1, keypoint2):
    """ Draws line between two keypoints onto image and returns annotated image. """
    cv2.line(image, keypoint1, keypoint2, (255, 255, 255), 4)

def draw_vector(image, start_point, vector, color=(255, 255, 255), thickness=4):
    end_point = tuple(np.array(start_point) + np.array(vector))
    cv2.line(image, tuple(start_point), end_point, color, thickness)

def get_mainpoint(left, right, part):
    """ For each important part (eg. hip), if both left and right part exists, we get midpoint. If only left or right exist, then we set the point to left or right. """
    main_point = (0,0)
    if left != (0,0) and right != (0,0):
        # print(f'both left and right {part} detected')
        main_point = calculate_midpoint(left, right)
    elif left != (0,0):
        # print(f'only left {part} detected')
        main_point = left
    elif right != (0,0):
        # print(f'only right {part} detected')
        shoulder = right
    return main_point

def calculate_angle_with_x_axis(vector):
    x, y = vector
    angle_rad = math.atan2(y, x)
    angle_deg = math.degrees(angle_rad)
    return angle_deg

def calculate_vector(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    vector_x = x2 - x1
    vector_y = y2 - y1
    return (vector_x, vector_y)

def calculate_angle(keypoint1, keypoint2):
    x1, y1 = keypoint1
    x2, y2 = keypoint2
    angle_rad = math.atan2(y2 - y1, x2 - x1)
    angle_deg = math.degrees(angle_rad)
    return angle_deg

def action_state(spine_leg_theta, spine_x_axis_phi):
    # check angle validity (should be 0/positive)
    if spine_leg_theta > -1 and spine_x_axis_phi > -1:
        # stand
        if spine_leg_theta>120 and spine_x_axis_phi>25:
            return "standing"
        # sit
        elif spine_leg_theta<120 and spine_x_axis_phi>25:
            return "sitting"
        # lying down
        elif spine_x_axis_phi<=25:
            return "lying down"
        # others?
    else:
        raise Exception("invalid theta/phi angles")
    