# import the necessary packages
from ultralytics import YOLO
import utils
import my_defs
import numpy as np

class YoloPoseDetect:
    def __init__(self, conf_threshold=0.5, debug=False):
        self.debug = debug
        self.conf_threshold = conf_threshold
        self.model = None
    
    def set_model(self, model_path=None):
        """ Set YOLO pose model. """
        if model_path:
            self.model = YOLO(model_path)    
        else:
            raise Exception("Yolo pose model path not specified.")
        
    def yolo_predict(self, frame):
        """ 
        Run inference on frame using YOLO pose model.
        
        """
        results = self.model.track(frame, conf=0.5, verbose=False, persist=True, tracker="bytetrack.yaml")
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
                boxes = result.boxes.xywh.tolist() # bounding boxes  
                # if keypts detected
                if num_pts !=0:
                    pass
    
    
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
        shoulder = utils.get_mainpoint(keypts_dict['left_shoulder']['xy'], keypts_dict['right_shoulder']['xy'], 
                                        keypts_dict['left_shoulder']['conf_score'], keypts_dict['right_shoulder']['conf_score'], 
                                        conf_threshold=self.conf_threshold,  part = "shoulders")
        hips = utils.get_mainpoint(keypts_dict['left_hip']['xy'], keypts_dict['right_hip']['xy'], keypts_dict['left_hip']['conf_score'], 
                                    keypts_dict['right_hip']['conf_score'], 
                                    conf_threshold=self.conf_threshold, part = "hips")
        knees = utils.get_mainpoint(keypts_dict['left_knee']['xy'], keypts_dict['right_knee']['xy'], keypts_dict['left_knee']['conf_score'], 
                                    keypts_dict['right_knee']['conf_score'], 
                                    conf_threshold=self.conf_threshold, part = "knees")
        ankles = utils.get_mainpoint(keypts_dict['left_ankle']['xy'], keypts_dict['right_ankle']['xy'], keypts_dict['left_ankle']['conf_score'], 
                                    keypts_dict['right_ankle']['conf_score'], 
                                    conf_threshold=self.conf_threshold, part = "ankles")    
        
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
        
        if legs_vector and ankle_vector:
            knee_x_axis = utils.calculate_vector(knees, (knees[0]+20, knees[1]))
            ankle_beta = utils.angle_between(ankle_vector, knee_x_axis)
            
        if spine_leg_theta and spine_x_axis_phi and legs_y_axis_alpha and ankle_beta and spine_leg_ratio and self.debug:
            print(f'Theta {spine_leg_theta}, Phi: {spine_x_axis_phi}, Alpha: {legs_y_axis_alpha}, 
                  Beta: {ankle_beta}, Spine-Leg Ratio: {spine_leg_ratio}')
        
        state = None
        # if at least have phi, alpha, and ratio, then can determine state
        if spine_x_axis_phi and legs_y_axis_alpha and spine_leg_ratio:
            state = utils.determine_state(theta=spine_leg_theta, phi=spine_x_axis_phi, 
                                          alpha=legs_y_axis_alpha, beta=ankle_beta ,ratio=spine_leg_ratio)
            if self.debug:
                print(f'State: {state}')
            # cv2.putText(frame, state, (hips[0]+30, hips[1]+20),  cv2.FONT_HERSHEY_PLAIN,2,(155,200,0),2)
        
        # return these for storing fall detection data
        return spine_vector, legs_vector, hips, shoulder, state
    
    
    