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
"""Utility functions to display the pose detection results."""

import math
from typing import List, Tuple

import cv2
from Pose_Estimate.movenet.movenet_data import Person
import numpy as np
from utils import my_defs
from utils import pose_utils

# map edges to a RGB color
KEYPOINT_EDGE_INDS_TO_COLOR = {
    (0, 1): (147, 20, 255),
    (0, 2): (255, 255, 0),
    (1, 3): (147, 20, 255),
    (2, 4): (255, 255, 0),
    (0, 5): (147, 20, 255),
    (0, 6): (255, 255, 0),
    (5, 7): (147, 20, 255),
    (7, 9): (147, 20, 255),
    (6, 8): (255, 255, 0),
    (8, 10): (255, 255, 0),
    (5, 6): (0, 255, 255),
    (5, 11): (147, 20, 255),
    (6, 12): (255, 255, 0),
    (11, 12): (0, 255, 255),
    (11, 13): (147, 20, 255),
    (13, 15): (147, 20, 255),
    (12, 14): (255, 255, 0),
    (14, 16): (255, 255, 0)
}

# A list of distictive colors
COLOR_LIST = [
    (47, 79, 79),
    (139, 69, 19),
    (0, 128, 0),
    (0, 0, 139),
    (255, 0, 0),
    (255, 215, 0),
    (0, 255, 0),
    (0, 255, 255),
    (255, 0, 255),
    (30, 144, 255),
    (255, 228, 181),
    (255, 105, 180),
]


def visualize(
    image: np.ndarray,
    list_persons: List[Person],
    keypoint_color: Tuple[int, ...] = None,
    keypoint_threshold: float = 0.05,
    instance_threshold: float = 0.1,
) -> np.ndarray:
  """Draws landmarks and edges on the input image and return it.

  Args:
    image: The input RGB image.
    list_persons: The list of all "Person" entities to be visualize.
    keypoint_color: the colors in which the landmarks should be plotted.
    keypoint_threshold: minimum confidence score for a keypoint to be drawn.
    instance_threshold: minimum confidence score for a person to be drawn.

  Returns:
    Image with keypoints and edges.
  """
  for person in list_persons:
    if not person:
        print("No person detected.")
        break
    if person.score < instance_threshold:
      continue

    keypoints = person.keypoints
    bounding_box = person.bounding_box

    # Assign a color to visualize keypoints.
    if keypoint_color is None:
      if person.id is None:
        # If there's no person id, which means no tracker is enabled, use
        # a default color.
        person_color = (0, 255, 0)
      else:
        # If there's a person id, use different color for each person.
        person_color = COLOR_LIST[person.id % len(COLOR_LIST)]
    else:
      person_color = keypoint_color

    # Draw all the landmarks
    for i in range(len(keypoints)):
      if keypoints[i].score >= keypoint_threshold:
        cv2.circle(image, keypoints[i].coordinate, 2, person_color, 4)

    # Draw all the edges
    for edge_pair, edge_color in KEYPOINT_EDGE_INDS_TO_COLOR.items():
      if (keypoints[edge_pair[0]].score > keypoint_threshold and
          keypoints[edge_pair[1]].score > keypoint_threshold):
        cv2.line(image, keypoints[edge_pair[0]].coordinate,
                 keypoints[edge_pair[1]].coordinate, edge_color, 2)

    # Draw bounding_box with multipose
    if bounding_box is not None:
      start_point = bounding_box.start_point
      end_point = bounding_box.end_point
      cv2.rectangle(image, start_point, end_point, person_color, 2)
      # Draw id text when tracker is enabled for MoveNet MultiPose model.
      # (id = None when using single pose model or when tracker is None)
      if person.id:
        id_text = 'id = ' + str(person.id)
        cv2.putText(image, id_text, start_point, cv2.FONT_HERSHEY_PLAIN, 1,
                    (0, 0, 255), 1)

  return image


def keep_aspect_ratio_resizer(
    image: np.ndarray, target_size: int) -> Tuple[np.ndarray, Tuple[int, int]]:
  """Resizes the image.

  The function resizes the image such that its longer side matches the required
  target_size while keeping the image aspect ratio. Note that the resizes image
  is padded such that both height and width are a multiple of 32, which is
  required by the model. See
  https://tfhub.dev/google/tfjs-model/movenet/multipose/lightning/1 for more
  detail.

  Args:
    image: The input RGB image as a numpy array of shape [height, width, 3].
    target_size: Desired size that the image should be resize to.

  Returns:
    image: The resized image.
    (target_height, target_width): The actual image size after resize.

  """
  height, width, _ = image.shape
  if height > width:
    scale = float(target_size / height)
    target_height = target_size
    scaled_width = math.ceil(width * scale)
    image = cv2.resize(image, (scaled_width, target_height))
    target_width = int(math.ceil(scaled_width / 32) * 32)
  else:
    scale = float(target_size / width)
    target_width = target_size
    scaled_height = math.ceil(height * scale)
    image = cv2.resize(image, (target_width, scaled_height))
    target_height = int(math.ceil(scaled_height / 32) * 32)

  padding_top, padding_left = 0, 0
  padding_bottom = target_height - image.shape[0]
  padding_right = target_width - image.shape[1]
  # add padding to image
  image = cv2.copyMakeBorder(image, padding_top, padding_bottom, padding_left,
                             padding_right, cv2.BORDER_CONSTANT)
  return image, (target_height, target_width)



def update_data(list_persons, prev_data, frame, curr_time, debug = False):
    """ Update prev_data with new data from list_persons."""
    for person in list_persons:
      if not person:
        continue
      if person.id:
        id = person.id
      else:
        id = 0
      spine_vector, legs_vector, hips, shoulders, state = extract_keypts(person, frame)
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
  
def extract_keypts(person, frame, conf_threshold=0.2, debug=False):
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