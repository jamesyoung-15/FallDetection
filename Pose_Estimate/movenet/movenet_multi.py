# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Code to run a pose estimation with a TFLite Movenet_multipose model."""

import os
import time
from typing import List

import cv2
from Pose_Estimate.movenet.movenet_data import BodyPart
from Pose_Estimate.movenet.movenet_data import KeyPoint
from Pose_Estimate.movenet.movenet_data import Person
from Pose_Estimate.movenet.movenet_data import Point
from Pose_Estimate.movenet.movenet_data import Rectangle
import numpy as np
from Pose_Estimate.movenet.bounding_box_tracker import BoundingBoxTracker
from Pose_Estimate.movenet.keypoint_tracker import KeypointTracker
from Pose_Estimate.movenet.movenet_tracker import TrackerConfig
import Pose_Estimate.movenet.movenet_utils as movenet_utils
from utils import my_defs
from utils import pose_utils

# pylint: disable=g-import-not-at-top
try:
  # Import TFLite interpreter from tflite_runtime package if it's available.
  from tflite_runtime.interpreter import Interpreter
except ImportError:
  # If not, fallback to use the TFLite interpreter from the full TF package.
  import tensorflow as tf
  Interpreter = tf.lite.Interpreter


class MoveNetMultiPose(object):
  """A wrapper class for a MultiPose TFLite pose estimation model."""

  def __init__(self,
               model_name: str,
               tracker_type: str = 'bounding_box',
               input_size: int = 256) -> None:
    """Initialize a MultiPose pose estimation model.

    Args:
      model_name: Name of the TFLite multipose model.
      tracker_type: Type of Tracker('keypoint' or 'bounding_box')
      input_size: Size of the longer dimension of the input image.
    """
    # Append .tflite extension to model_name if there's no extension.
    _, ext = os.path.splitext(model_name)
    if not ext:
      model_name += '.tflite'

    # Store the input size parameter.
    self._input_size = input_size

    # Initialize the TFLite model.
    interpreter = Interpreter(model_path=model_name, num_threads=4)

    self._input_details = interpreter.get_input_details()
    self._output_details = interpreter.get_output_details()
    self._input_type = self._input_details[0]['dtype']

    self._input_height = interpreter.get_input_details()[0]['shape'][1]
    self._input_width = interpreter.get_input_details()[0]['shape'][2]

    self._interpreter = interpreter

    # Initialize a tracker.
    config = TrackerConfig()
    if tracker_type == 'keypoint':
      self._tracker = KeypointTracker(config)
    elif tracker_type == 'bounding_box':
      self._tracker = BoundingBoxTracker(config)
    else:
      print('ERROR: Tracker type {0} not supported. No tracker will be used.'
            .format(tracker_type))
      self._tracker = None

  def detect(self,
             input_image: np.ndarray,
             detection_threshold: float = 0.2) -> List[Person]:
    """Run detection on an input image.

    Args:
      input_image: A [height, width, 3] RGB image. Note that height and width
        can be anything since the image will be immediately resized according to
        the needs of the model within this function.
      detection_threshold: minimum confidence score for an detected pose to be
        considered.

    Returns:
      A list of Person instances detected from the input image.
    """

    is_dynamic_shape_model = self._input_details[0]['shape_signature'][2] == -1
    # Resize and pad the image to keep the aspect ratio and fit the expected
    # size.
    if is_dynamic_shape_model:
      resized_image, _ = movenet_utils.keep_aspect_ratio_resizer(
          input_image, self._input_size)
      input_tensor = np.expand_dims(resized_image, axis=0)
      self._interpreter.resize_tensor_input(
          self._input_details[0]['index'], input_tensor.shape, strict=True)
    else:
      resized_image = cv2.resize(input_image,
                                 (self._input_width, self._input_height))
      input_tensor = np.expand_dims(resized_image, axis=0)
    self._interpreter.allocate_tensors()

    # Run inference with the MoveNet MultiPose model.
    self._interpreter.set_tensor(self._input_details[0]['index'],
                                 input_tensor.astype(self._input_type))
    self._interpreter.invoke()

    # Get the model output
    model_output = self._interpreter.get_tensor(
        self._output_details[0]['index'])

    image_height, image_width, _ = input_image.shape
    return self._postprocess(model_output, image_height, image_width,
                             detection_threshold)

  def _postprocess(self, keypoints_with_scores: np.ndarray, image_height: int,
                   image_width: int,
                   detection_threshold: float) -> List[Person]:
    """Returns a list "Person" corresponding to the input image.

    Note that coordinates are expressed in (x, y) format for drawing
    utilities.

    Args:
      keypoints_with_scores: Output of the MultiPose TFLite model.
      image_height: height of the image in pixels.
      image_width: width of the image in pixels.
      detection_threshold: minimum confidence score for an entity to be
        considered.

    Returns:
      A list of Person(keypoints, bounding_box, scores), each containing:
        * the coordinates of all keypoints of the detected entity;
        * the bounding boxes of the entity.
        * the confidence core of the entity.
    """

    _, num_instances, _ = keypoints_with_scores.shape
    list_persons = []
    keypts_dict = {}
    for idx in range(num_instances):
      # Skip a detected pose if its confidence score is below the threshold
      person_score = keypoints_with_scores[0, idx, 55]
      if person_score < detection_threshold:
        continue

      # Extract the keypoint coordinates and scores
      kpts_y = keypoints_with_scores[0, idx, range(0, 51, 3)]
      kpts_x = keypoints_with_scores[0, idx, range(1, 51, 3)]
      scores = keypoints_with_scores[0, idx, range(2, 51, 3)]
      
      # Create the list of keypoints
      keypoints = []
      for i in range(scores.shape[0]):
        keypoints.append(
            KeyPoint(
                BodyPart(i),
                Point(
                    int(kpts_x[i] * image_width),
                    int(kpts_y[i] * image_height)), scores[i]))
        
      # Calculate the bounding box
      rect = [
          keypoints_with_scores[0, idx, 51], keypoints_with_scores[0, idx, 52],
          keypoints_with_scores[0, idx, 53], keypoints_with_scores[0, idx, 54]
      ]
      bounding_box = Rectangle(
          Point(int(rect[1] * image_width), int(rect[0] * image_height)),
          Point(int(rect[3] * image_width), int(rect[2] * image_height)))

      # Create a Person instance corresponding to the detected entity.
      list_persons.append(Person(keypoints, bounding_box, person_score))
    if self._tracker:
      list_persons = self._tracker.apply(list_persons, time.time() * 1000)
    
    return list_persons
  
  def update_data(self, list_persons, prev_data, frame, curr_time, debug = False):
    """ Update prev_data with new data from list_persons."""
    for person in list_persons:
      if not person:
        continue
      id = person.id
      spine_vector, legs_vector, hips, shoulders, state = self.extract_keypts(person, frame)
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
          if debug:
              print(f"ID {key} hasn't checked over {time_till_delete} seconds")
              print(f'Deleting ID {key}')
          del prev_data[key]
  
  def extract_keypts(self, person, frame, conf_threshold=0.2, debug=False):
    """ Extract shoulder, hips, knees, and ankles keypts from movenet multipose model."""

    # extract relevant keypoints and confidence scores into nested dict
    keypts_dict = {}
    for part in my_defs.IMPORTANT_PTS:
        keypts_dict[part] = {}
        keypts_dict[part]['xy'] = (int(person.keypoints[my_defs.KEYPOINT_DICT[part]].coordinate.x), 
                                   int(person.keypoints[my_defs.KEYPOINT_DICT[part]].coordinate.y))
        keypts_dict[part]['conf_score'] = person.keypoints[my_defs.KEYPOINT_DICT[part]].score
    
    # check whether left/right keypt exist, if both exist get midpoint
    shoulder = pose_utils.get_mainpoint(keypts_dict['left_shoulder']['xy'], keypts_dict['right_shoulder']['xy'], 
                                    keypts_dict['left_shoulder']['conf_score'], keypts_dict['right_shoulder']['conf_score'], 
                                    conf_threshold=conf_threshold,  part = "shoulders")
    hips = pose_utils.get_mainpoint(keypts_dict['left_hip']['xy'], keypts_dict['right_hip']['xy'], keypts_dict['left_hip']['conf_score'], 
                                keypts_dict['right_hip']['conf_score'], 
                                conf_threshold=conf_threshold, part = "hips")
    knees = pose_utils.get_mainpoint(keypts_dict['left_knee']['xy'], keypts_dict['right_knee']['xy'], keypts_dict['left_knee']['conf_score'], 
                                keypts_dict['right_knee']['conf_score'], 
                                conf_threshold=conf_threshold, part = "knees")
    ankles = pose_utils.get_mainpoint(keypts_dict['left_ankle']['xy'], keypts_dict['right_ankle']['xy'], keypts_dict['left_ankle']['conf_score'], 
                                keypts_dict['right_ankle']['conf_score'], 
                                conf_threshold=conf_threshold, part = "ankles")    

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
        
    if spine_leg_theta and spine_x_axis_phi and legs_y_axis_alpha and ankle_beta and spine_leg_ratio and debug:
        print(f'Theta {spine_leg_theta}, Phi: {spine_x_axis_phi}, Alpha: {legs_y_axis_alpha}, Beta: {ankle_beta}, Spine-Leg Ratio: {spine_leg_ratio}')

    state = None
    # if at least have phi, alpha, and ratio, then can determine state
    if spine_x_axis_phi and legs_y_axis_alpha and spine_leg_ratio:
        state = pose_utils.determine_state(theta=spine_leg_theta, phi=spine_x_axis_phi, 
                                      alpha=legs_y_axis_alpha, beta=ankle_beta ,ratio=spine_leg_ratio)
        if debug:
            print(f'State: {state}')
        # cv2.putText(frame, state, (hips[0]+30, hips[1]+20),  cv2.FONT_HERSHEY_PLAIN,2,(155,200,0),2)
        
    # return these for storing fall detection data
    return spine_vector, legs_vector, hips, shoulder, state