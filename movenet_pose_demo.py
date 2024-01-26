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
"""Main script to run pose classification and pose estimation."""
import argparse
import logging
import sys
import time

import cv2
from Pose_Estimate.movenet.movenet_single import Movenet
from Pose_Estimate.movenet.movenet_multi import MoveNetMultiPose
# from ml import Posenet
import Pose_Estimate.movenet.movenet_utils as movenet_utils


def run(estimation_model: str, tracker_type: str, media_src: str, width: int, height: int) -> None:
  """Continuously run inference on images acquired from the camera.

  Args:
    estimation_model: Name of the TFLite pose estimation model.
    tracker_type: Type of Tracker('keypoint' or 'bounding_box').
    media_src: The camera id to be passed to OpenCV.
    width: The width of the frame captured from the camera.
    height: The height of the frame captured from the camera.
  """

  # Notify users that tracker is only enabled for MoveNet MultiPose model.
  if tracker_type and ('movenet_multipose' not in estimation_model):
    logging.warning(
        'No tracker will be used as tracker can only be enabled for '
        'MoveNet MultiPose model.')

  # Initialize the pose estimator selected.
  # estimation_model = 'models/tflite/' + estimation_model + '.tflite'
  if 'movenet_singlepose' in estimation_model:
    print("Using Movenet Singlepose (Lightning)")
    pose_detector = Movenet(estimation_model)
  elif estimation_model == 'posenet':
    # pose_detector = Posenet(estimation_model)
    pass
  elif 'movenet_multipose' in estimation_model:
    print("Using MoveNet MultiPose")
    pose_detector = MoveNetMultiPose(estimation_model, tracker_type)
  else:
    sys.exit('ERROR: Model is not supported.')

  # Variables to calculate FPS
  counter, fps = 0, 0
  start_time = time.time()

  # Start capturing video input from the camera
  cap = cv2.VideoCapture(media_src)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

  # Visualization parameters
  row_size = 20  # pixels
  left_margin = 24  # pixels
  text_color = (0, 0, 255)  # red
  font_size = 1
  font_thickness = 1
  fps_avg_frame_count = 10
  interval = 5
  num_frame = 0

  # Continuously capture images from the camera and run inference
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      sys.exit(
          'ERROR: Unable to read from webcam. Please verify your webcam settings.'
      )
    num_frame += 1
    counter += 1
    # image = cv2.flip(image, 1)

    if num_frame >= interval:
      num_frame = 0
      if 'movenet_multipose' in estimation_model:
        # Run pose estimation using a MultiPose model.
        list_persons = pose_detector.detect(image)
      else:
        # Run pose estimation using a SinglePose model, and wrap the result in an
        # array.
        list_persons = [pose_detector.detect(image)]
      if list_persons:
        print(list_persons[0])
      # Draw keypoints and edges on input image
      image = movenet_utils.visualize(image, list_persons)


    # Calculate the FPS
    if counter % fps_avg_frame_count == 0:
      end_time = time.time()
      fps = fps_avg_frame_count / (end_time - start_time)
      start_time = time.time()

    # Show the FPS
    fps_text = 'FPS = ' + str(int(fps))
    text_location = (left_margin, row_size)
    cv2.putText(image, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                font_size, text_color, font_thickness)

    # Stop the program if the ESC key is pressed.
    if cv2.waitKey(1) == 27:
      break
    cv2.imshow(estimation_model, image)

  cap.release()
  cv2.destroyAllWindows()


def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '--model',
      help='Name of estimation model.',
      required=False,
      default='models/tflite/movenet_multipose.tflite')
  parser.add_argument(
      '--tracker',
      help='Type of tracker to track poses across frames.',
      required=False,
      default='bounding_box')
  parser.add_argument(
      '--src', help='Id of camera.', required=False, default=0)
  parser.add_argument(
      '--width',
      help='Width of frame to capture from camera.',
      required=False,
      type=int,
      default=480)
  parser.add_argument(
      '--height',
      help='Height of frame to capture from camera.',
      required=False,
      type=int,
      default=480)
  args = parser.parse_args()

  run(args.model, args.tracker, args.src, args.width, args.height)


if __name__ == '__main__':
  main()