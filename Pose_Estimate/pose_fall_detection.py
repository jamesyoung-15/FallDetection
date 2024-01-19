import numpy as np
from utils import pose_utils

class PoseFallDetector:
    def __init__(self, debug=False):
        self.debug = debug
        # self.data_buf = {}
    
    def fall_detection(self, prev_data, frame_width=480, frame_height=480):
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
                for i in range(num_frames):
                    for j in range(i+1, num_frames):
                        # calculate shoulder and hip pixel y value differences (normalized to frame height)
                        print(f'Shoulder diff: {value["shoulders"][i][1] - value["shoulders"][j][1]}')
                        print(f'Shoulder diff norm: {value["shoulders"][i][1]/frame_height - value["shoulders"][j][1]/frame_height}')
                        print(f'Shoulder diff: {value["hips"][i][1] - value["hips"][j][1]}')
                        print(f'Shoulder diff norm: {value["hips"][i][1]/frame_height - value["hips"][j][1]/frame_height}')
                        shoulder_diffs.append(value["shoulders"][i][1] - value["shoulders"][j][1])
                        hip_diffs.append(abs(value["hips"][i][1] - value["hips"][j][1]))
                        spine_angles.append(pose_utils.angle_between(value['spine_vector'][i], value['spine_vector'][j]))
                
                
                min_shoulder_diff = min(shoulder_diffs)
                max_hip_diff = max(hip_diffs)
                max_angle = max(spine_angles)
                
                state = prev_data[key]['state'][-1]
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
                elif 25<=max_angle<60:
                    angle_conf = 1
                else:
                    angle_conf = 2
                
                if max_hip_diff<15:
                    hip_conf = -1
                elif 15<=max_hip_diff<20:
                    hip_conf = 0
                elif 20<=max_hip_diff<50:
                    hip_conf = 1
                else:
                    hip_conf = 2
                
                if min_shoulder_diff>-5:
                    shoulder_conf = -1
                elif min_shoulder_diff>-15:
                    shoulder_conf = 0
                elif -15>=min_shoulder_diff>-40:
                    shoulder_conf = 1
                else:
                    shoulder_conf = 2
                    
                fall_conf = min(1,(angle_conf + hip_conf + shoulder_conf+ state_conf)/10)
                
                if self.debug and fall_conf>=0.3:
                    print(f'Confidence: {fall_conf}')
                    print(f'State: {state}')
                    print(f"Hip Diffs: {max_hip_diff}, Shoulder diffs: {min_shoulder_diff}")
                    print(f'angles: {spine_angles}')
                    print(f'spine: {prev_data[key]["spine_vector"]}')
                    print(f'hips: {prev_data[key]["hips"]}')
                    print()
                
                if fall_conf>=0.7:
                    fall_detected = True
                    print(f"\nHigh Probability of Person {key} Fall Detected!!")
                    print(f'Confidence: {fall_conf}')
                    # remove data from prev_data to avoid multiple detections
                    prev_data[key]['spine_vector'] = prev_data[key]['spine_vector'][3:]
                    prev_data[key]['hips'] = prev_data[key]['hips'][3:]
                    prev_data[key]['shoulders'] = prev_data[key]['shoulders'][3:]
                elif fall_conf<.7 and fall_conf>.5:
                    fall_detected = True
                    print(f'\Medium/Low Probability of Person {key} Fall Detected.')
        
        return fall_detected, fall_conf