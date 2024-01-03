import argparse
import math

def get_args():
    """ Parse arguments. Returns user/default set arguments """
    parser = argparse.ArgumentParser()
    # user settings
    parser.add_argument("--src", type=str, default='/dev/video0')
    parser.add_argument("--show", type=int, default=1)
    parser.add_argument("--width", help='cap width', type=int, default=640)
    parser.add_argument("--height", help='cap height', type=int, default=480)
    parser.add_argument("--conf_score", type=float, default=0.4)

    args = parser.parse_args()
    return args

def calculate_midpoint(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    midpoint_x = int((x1 + x2) / 2)
    midpoint_y = int((y1 + y2) / 2)
    return (midpoint_x, midpoint_y)

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