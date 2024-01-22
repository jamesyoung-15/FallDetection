# import the necessary packages
from ultralytics import YOLO
from utils import pose_utils
from utils import my_defs
import numpy as np
import math
import cv2

class YoloPoseDetector:
    def __init__(self, conf_threshold=0.5, debug=False, model_path="models/yolo-weights/yolov8n-pose.pt"):
        self.debug = debug
        self.conf_threshold = conf_threshold
        self.model = None
        self.model_path = model_path
    
    def load_model(self):
        """ Set YOLO pose model. """
        if self.model_path:
            self.model = YOLO(self.model_path)    
        else:
            raise Exception("Yolo pose model path not specified.")
        
    
        
    def yolo_predict(self, prev_data, frame, curr_time):
        """ 
        Run inference on frame using YOLO pose model, extract keypoints, determine state, and stores data to dict. Modifies prev_data input.
        
        Args:
        - prev_data: reference to dictionary, modifies dictionary appending data
        - frame: cv2 frame (numpy array), frame from video
        - curr_time: float, current time in seconds 
        
        """
        spine_vector, leg_vector, hips, shoulders, state = None, None, None, None, None
        results = self.model.track(frame, conf=0.5, verbose=False, persist=True, tracker="bytetrack.yaml")
        # get data from inference
        for result in results:
            keypts = result.keypoints
            # print(f'Keypoints: \n{kpts}')
            num_people = keypts.shape[0]
            num_pts = keypts.shape[1]
            track_id = [1, 1, 1] # random pad for testing
            boxes = None
            # frame = result.plot()
            if result.boxes.id is not None:
                track_id = result.boxes.id.tolist()
                boxes = result.boxes.xywh.tolist() # bounding boxes  
                # if keypts detected
                if num_pts !=0:
                    for i in range(num_people):
                        id = int(track_id[i])
                        spine_vector, leg_vector, hips, shoulders, state = self.extract_keypts(i, keypts, frame)
                        # append spine vector to prev_data
                        if spine_vector:
                            # create new entry if no existing data for person id in prev_data
                            if prev_data.get(id) == None:
                                prev_data[id] = {}
                                prev_data[id]['spine_vector'] = [spine_vector]
                                prev_data[id]['hips'] = [hips]
                                prev_data[id]['shoulders'] = [shoulders]
                                prev_data[id]['state'] = [state]
                            # otherwise append data, remove oldest data if more than 3 data points (3 frames)
                            else:
                                if len(prev_data[id]['spine_vector'])>=3:
                                    prev_data[id]['spine_vector'].pop(0)
                                    prev_data[id]['hips'].pop(0)
                                    prev_data[id]['shoulders'].pop(0)
                                    prev_data[id]['state'].pop(0)
                                prev_data[id]['spine_vector'].append(spine_vector)
                                prev_data[id]['hips'].append(hips)
                                prev_data[id]['shoulders'].append(shoulders)
                                prev_data[id]['state'].append(state)
                            # append inference time
                            prev_data[id]['last_check'] = curr_time
            # delete data if not checked for a while
            for key, value in prev_data.copy().items():
                time_till_delete = 2
                if curr_time -  value['last_check'] > time_till_delete:
                    if self.debug:
                        print(f"ID {key} hasn't checked over {time_till_delete} seconds")
                        print(f'Deleting ID {key}')
                    del prev_data[key]
        # return frame

    def get_xy(self,keypoint):
        """ Convert Yolo tensor keypoint data to array and returns (x,y)  """
        try:
            return int(keypoint[0].item()), int(keypoint[1].item())
        except:
            raise Exception("unable to get keypoint coordinate")

    def get_conf(self, conf_scores, part):
        """ Input conf_scores is keypoint.conf (list). Returns confidence score for specified keypoint (float). """
        try:
            return float(conf_scores[my_defs.KEYPOINT_DICT[part]])
        except:
            raise Exception("unable to get confidence score")
        
    def extract_keypts(self, person_num, keypts, frame):
        """ 
        Performs inference on each person in frame. Extracts the pose keypoints, checks the vector and angle 
        between points and limbs, and determines the state of the person (standing, sitting, fallen).
        
        Inputs:
        - person_num: int, index of person
        - keypts: tensor, keypoint data from yolo
        - frame: numpy array, frame from video
        
        Returns:
        - spine_vector: (x,y) array, vector between shoulder and hips
        - legs_vector: (x,y) array, vector between hips and knees
        - hips: (x,y) coordinate of hips
        - shoulder: (x,y) coordinate of shoulder
        - state: string, state of person
        """
        # extract relevant keypoints and confidence scores into nested dict
        keypts_dict = {}
        for parts in my_defs.IMPORTANT_PTS:
            keypts_dict[parts] = {}
            keypts_dict[parts]['xy'] = self.get_xy(keypts.xy[person_num, my_defs.KEYPOINT_DICT[parts]])
            keypts_dict[parts]['conf_score'] = self.get_conf(keypts.conf[person_num], parts)
        
        # check whether left/right keypt exist, if both exist get midpoint
        shoulder = pose_utils.get_mainpoint(keypts_dict['left_shoulder']['xy'], keypts_dict['right_shoulder']['xy'], 
                                        keypts_dict['left_shoulder']['conf_score'], keypts_dict['right_shoulder']['conf_score'], 
                                        conf_threshold=self.conf_threshold,  part = "shoulders")
        hips = pose_utils.get_mainpoint(keypts_dict['left_hip']['xy'], keypts_dict['right_hip']['xy'], keypts_dict['left_hip']['conf_score'], 
                                    keypts_dict['right_hip']['conf_score'], 
                                    conf_threshold=self.conf_threshold, part = "hips")
        knees = pose_utils.get_mainpoint(keypts_dict['left_knee']['xy'], keypts_dict['right_knee']['xy'], keypts_dict['left_knee']['conf_score'], 
                                    keypts_dict['right_knee']['conf_score'], 
                                    conf_threshold=self.conf_threshold, part = "knees")
        ankles = pose_utils.get_mainpoint(keypts_dict['left_ankle']['xy'], keypts_dict['right_ankle']['xy'], keypts_dict['left_ankle']['conf_score'], 
                                    keypts_dict['right_ankle']['conf_score'], 
                                    conf_threshold=self.conf_threshold, part = "ankles")    
        
        # if relevant keypt exist draw pt
        if shoulder:
            pose_utils.draw_keypoint(frame, shoulder)
            # print(f'Shoulder: {shoulder}')
        if hips:
            pose_utils.draw_keypoint(frame, hips)
            # print(f'Hips: {hips}')
        if knees:
            pose_utils.draw_keypoint(frame, knees)
            # print(f'Knees: {knees}')
        
        if ankles:
            pose_utils.draw_keypoint(frame, ankles)
            # print(f'Ankles: {ankles}')
            
        # if keypts exist draw line to connect them, calculate vector
        spine_vector = None
        legs_vector = None
        ankle_vector = None
        spine_vector_length = None
        legs_vector_length = None
        # spine vector
        if shoulder and hips:
            spine_vector = pose_utils.calculate_vector(hips, shoulder)
            # pose_utils.draw_keypoint_line(frame, shoulder, hips)
            pose_utils.draw_vector(frame, hips, spine_vector)
            # print(f'Spine Vector: {spine_vector}')
            spine_vector_length = np.linalg.norm(spine_vector)
        
        # leg vector
        if hips and knees:
            legs_vector = pose_utils.calculate_vector(hips, knees)
            # legs_vector = pose_utils.calculate_vector(knees, hips)
            # pose_utils.draw_keypoint_line(frame, hips, knees)
            pose_utils.draw_vector(frame, hips, legs_vector)
            # print(f'Leg Vector: {legs_vector}')
            legs_vector_length = np.linalg.norm(legs_vector)
        
        # ankle vector
        if knees and ankles:
            ankle_vector = pose_utils.calculate_vector(knees, ankles)
            pose_utils.draw_vector(frame, knees, ankle_vector)
        
        # spine-leg ratio
        if spine_vector_length is not None and legs_vector_length is not None:
            spine_leg_ratio = spine_vector_length/legs_vector_length
        
        # calculate vector if main pts exist
        spine_leg_theta = None # angle between spine (vector between shoulder and hips) and legs (vector between hips and knees)
        spine_x_axis_phi = None # angle between spine (vector between shoulder and hips) and x_axis along hip point
        legs_y_axis_alpha = None # angle between legs (vector between hips and knees) and y_axis along hip point
        ankle_beta = None # angle between ankle and x_axis along knee point
        if spine_vector and legs_vector:
            spine_leg_theta = pose_utils.angle_between(spine_vector, legs_vector)
            hips_x_axis = pose_utils.calculate_vector(hips, (hips[0]+20, hips[1]))
            hips_y_axis = pose_utils.calculate_vector(hips, (hips[0], hips[1]+20))
            spine_x_axis_phi = pose_utils.angle_between(spine_vector, hips_x_axis)
            legs_y_axis_alpha = pose_utils.angle_between(legs_vector, hips_y_axis)
        
        if legs_vector and ankle_vector:
            knee_x_axis = pose_utils.calculate_vector(knees, (knees[0]+20, knees[1]))
            ankle_beta = pose_utils.angle_between(ankle_vector, knee_x_axis)
            
        if spine_leg_theta and spine_x_axis_phi and legs_y_axis_alpha and ankle_beta and spine_leg_ratio and self.debug:
            print(f'Theta {spine_leg_theta}, Phi: {spine_x_axis_phi}, Alpha: {legs_y_axis_alpha}, Beta: {ankle_beta}, Spine-Leg Ratio: {spine_leg_ratio}')
        
        state = None
        # if at least have phi, alpha, and ratio, then can determine state
        if spine_x_axis_phi and legs_y_axis_alpha and spine_leg_ratio:
            state = pose_utils.determine_state(theta=spine_leg_theta, phi=spine_x_axis_phi, 
                                          alpha=legs_y_axis_alpha, beta=ankle_beta ,ratio=spine_leg_ratio)
            if self.debug:
                print(f'State: {state}')
            # cv2.putText(frame, state, (hips[0]+30, hips[1]+20),  cv2.FONT_HERSHEY_PLAIN,2,(155,200,0),2)
            
        # return these for storing fall detection data
        return spine_vector, legs_vector, hips, shoulder, state
    
    
    
    