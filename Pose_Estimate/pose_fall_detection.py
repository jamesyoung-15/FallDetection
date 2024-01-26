import numpy as np
from utils import pose_utils
import math

class PoseFallDetector:
    def __init__(self, debug=False):
        self.debug = debug
        # self.data_buf = {}
    
    def fall_detection_v2(self, prev_data, frame_width=480, frame_height=480):
        """ 
        Fall detection algorithm using previous 3 frame data.
        
        Args:
        - prev_data: dictionary of previous frame data
        
        Return:
        - fall_detected: boolean indicating whether fall detected
        - fall_conf: confidence score of fall detection (still tweaking)
        
        """
        fall_detected = False
        fall_conf = 0
        # fall detection using previous frames
        for key, value in prev_data.copy().items():
            # detect fall from spine vector in past 3 frames
            num_frames = len(value['spine_vector'])
            if num_frames == 3:
                shoulder_diffs = []
                hip_diffs = []
                spine_angles = []
                for i in range(num_frames-1, -1, -1):
                    for j in range(i-1, -1, -1):
                        shoulder_diff_y = (value["shoulders"][i][1] - value["shoulders"][j][1])/frame_height
                        hip_diff_y = (value["hips"][i][1] - value["hips"][j][1])/frame_height
                        shoulder_diffs.append(shoulder_diff_y)
                        hip_diffs.append(hip_diff_y)
                        spine_angles.append(pose_utils.angle_between(value['spine_vector'][j], value['spine_vector'][i]))
                
                
                max_shoulder_diff = max(shoulder_diffs)
                max_hip_diff = max(hip_diffs)
                max_angle = max(spine_angles)
                state = prev_data[key]['state'][-1]
                fall_conf = self.get_conf(state, max_angle, max_hip_diff, max_shoulder_diff)
                
                if self.debug and fall_conf>=0.3:
                    print(f'Confidence: {fall_conf}')
                    print(f'State: {state}')
                    print(f"Hip Diffs: {max_hip_diff}, Shoulder diffs: {max_shoulder_diff}")
                    print(f'angles: {spine_angles}')
                    print(f'spine: {prev_data[key]["spine_vector"]}')
                    print(f'hips: {prev_data[key]["hips"]}, shoulders: {prev_data[key]["shoulders"]}, frame height: {frame_height}')
                    print()
                
                if fall_conf>=0.7:
                    fall_detected = True
                    print(f"\nHigh Probability of Person {key} Fall Detected!!")
                    print(f'Confidence: {fall_conf}')
                    # remove data from prev_data to avoid multiple detections
                    prev_data[key]['spine_vector'] = prev_data[key]['spine_vector'][3:]
                    prev_data[key]['hips'] = prev_data[key]['hips'][3:]
                    prev_data[key]['shoulders'] = prev_data[key]['shoulders'][3:]
                    # exit loop if at least one person has high probability of fall
                    break
                    
                elif fall_conf<.7 and fall_conf>.5:
                    fall_detected = True
                    print(f'\nMedium/Low Probability of Person {key} Fall Detected.')
        
        return fall_detected, fall_conf
    
    def get_conf(self, state, max_angle, max_hip_diff, max_shoulder_diff):
        if state == "standing":
            state_conf = -2
        elif state == "sitting":
            state_conf = 2
        elif state == "lying down":
            state_conf = 3
        else:
            state_conf = 0
        
        if max_angle<25:
            angle_conf = -2
        elif 25<=max_angle<50:
            angle_conf = 1
        else:
            angle_conf = 2
        
        if max_hip_diff<=0:
            hip_conf = -1
        elif 0<max_hip_diff<0.01:
            hip_conf = 0
        elif 0.01<=max_hip_diff<0.1:
            hip_conf = 1
        else:
            hip_conf = 2
        
        if max_shoulder_diff<=0:
            shoulder_conf = -1
        elif 0<max_shoulder_diff<0.01:
            shoulder_conf = 0
        elif 0.01<=max_shoulder_diff<0.1:
            shoulder_conf = 1
        else:
            shoulder_conf = 2
        return min(1,(angle_conf + hip_conf + shoulder_conf+ state_conf)/10)
    