import cv2
import mediapipe as mp
import numpy as np
import time
import math

class PoseModule:
    def __init__(self, mode=False, detection_confidence = 0.5, tracking_confidence=0.5 , smooth=True) -> None:
        """ Init class. Setup mediapipe variables """
        # setup media pipe
        self.mode = mode
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence
        self.pose = self.mp_pose.Pose(min_detection_confidence=self.detection_confidence, min_tracking_confidence=self.tracking_confidence)
        # store detected pose results
        self.results = None
        
        
    def find_pose(self, img, draw=True):
        """ Performs pose estimate with mediapipe. Returns passed image, if draw is set to true will annotate image with landmarks """
        # convert to RGB
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        
        if self.results.pose_landmarks:
            if draw:
                self.mp_draw.draw_landmarks(img, self.results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
        
        return img, self.results
    
    def get_landmarks(self, img, draw=True):
        """ Return landmark info from mediapipe """
        landmarks = []
        if self.results.pose_landmarks:
            for landmark_id, landmark in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(landmark.x*w), int(landmark.y*h)
                landmarks.append([landmark_id, cx, cy])
                if draw:
                    cv2.circle(img,(cx,cy), 5, (255,0,0),cv2.FILLED)
        return landmarks
    
    def get_angle(self, landmark1, landmark2):
        """ Calculate angle between 2 landmarks """
        # Convert landmark positions to 2D vectors
        vec1 = mp.solutions.pose.PoseLandmark(landmark1).x, mp.solutions.pose.PoseLandmark(landmark1).y
        vec2 = mp.solutions.pose.PoseLandmark(landmark2).x, mp.solutions.pose.PoseLandmark(landmark2).y

        # Calculate the vertical distance between the landmarks
        vertical_distance = vec2[1] - vec1[1]

        # Calculate the horizontal distance between the landmarks
        horizontal_distance = vec2[0] - vec1[0]

        # Calculate the angle in radians
        angle_rad = math.atan2(vertical_distance, horizontal_distance)

        # Convert the angle to degrees
        angle_deg = math.degrees(angle_rad)

        return angle_deg
    
    def get_angle(self, landmark1, landmark2, landmark3):
        """ Calculate angle between 3 landmarks """
        # Convert landmark positions to 2D vectors
        vec1 = mp.solutions.pose.PoseLandmark(landmark1).x, mp.solutions.pose.PoseLandmark(landmark1).y
        vec2 = mp.solutions.pose.PoseLandmark(landmark2).x, mp.solutions.pose.PoseLandmark(landmark2).y
        vec3 = mp.solutions.pose.PoseLandmark(landmark3).x, mp.solutions.pose.PoseLandmark(landmark3).y

        # Calculate vectors between the landmarks
        v1 = (vec1[0] - vec2[0], vec1[1] - vec2[1])
        v2 = (vec3[0] - vec2[0], vec3[1] - vec2[1])

        # Calculate the dot product of the vectors
        dot_product = v1[0] * v2[0] + v1[1] * v2[1]

        # Calculate the magnitudes of the vectors
        magnitude_v1 = math.sqrt(v1[0]**2 + v1[1]**2)
        magnitude_v2 = math.sqrt(v2[0]**2 + v2[1]**2)

        # Calculate the cosine of the angle between the vectors
        cosine_angle = dot_product / (magnitude_v1 * magnitude_v2)

        # Calculate the angle in radians
        angle_rad = math.acos(cosine_angle)

        # Convert the angle to degrees
        angle_deg = math.degrees(angle_rad)

        return angle_deg
    
    def get_distance():
        """ Get euclidean distance between landmarks """
        pass