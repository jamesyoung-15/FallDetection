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
# from Pose_Estimate.movenet.movenet_single import Movenet
from Pose_Estimate.movenet.movenet_multi import MoveNetMultiPose
# from ml import Posenet
import Pose_Estimate.movenet.movenet_utils as movenet_utils
from Pose_Estimate.pose_fall_detection import PoseFallDetector

def run(estimation_model: str, tracker_type: str, 
        media_src: str, vid_width: int, vid_height: int, delay: int = 1,
        to_show: bool = True, interval: int = 5, debug: bool = False,
        fps: int = 24, save_video: bool = False, benchmark: bool = True) -> None:
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
  # if 'movenet_singlepose' in estimation_model:
  #   print("Using Movenet Singlepose (Lightning)")
  #   pose_detector = Movenet(estimation_model)
  # elif estimation_model == 'posenet':
  #   # pose_detector = Posenet(estimation_model)
  #   pass
  # elif 'movenet_multipose' in estimation_model:
  #   print("Using MoveNet MultiPose")
  #   pose_detector = MoveNetMultiPose(estimation_model, tracker_type)
  # else:
  #   sys.exit('ERROR: Model is not supported.')
	
	# init
	pose_detector = MoveNetMultiPose(estimation_model, tracker_type)
	fall_detector = PoseFallDetector(debug=False)

	# Variables to calculate FPS
	counter, fps_track = 0, 0
	start_time, start_time_fps = time.time(), time.time()

	is_webcam = False
	if '/dev/video' in media_src:
		is_webcam = True

	# non threaded video stream
	cap = cv2.VideoCapture(media_src)
	# if video is webcam, set width and height, otherwise resizing with cap.set doesn't work
	if is_webcam:
		print("Detected using webcam...")
		print(f'Width: {vid_width}, Height: {vid_height}, FPS: {fps}')
		cap.set(cv2.CAP_PROP_FRAME_WIDTH, vid_width)
		cap.set(cv2.CAP_PROP_FRAME_HEIGHT, vid_height)
		cap.set(cv2.CAP_PROP_FPS, fps)
	elif not is_webcam:
		print("Detected non-webcam video...")
		vid_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
		vid_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	# get width and height
	print(f'Video Width: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)} , Video Height: {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}')
 
	if save_video == True:
		fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
		video_output = cv2.VideoWriter('demo.avi',  fourcc, 30, (int(cap.get(3)),int(cap.get(4)))) 

	# Visualization parameters
	row_size = 20  # pixels
	left_margin = 24  # pixels
	text_color = (0, 0, 255)  # red
	font_size = 1
	font_thickness = 1
	fps_avg_frame_count = 10

	# fall detection parameters
	num_frame = 0 # frame counter 
	prev_data = {} # dictionary to store prev frame data for determining action
	fall_detected, fall_conf = False, 0
	total_frames = 0

	# Continuously capture images from the camera and run inference
	while cap.isOpened():
		success, image = cap.read()
		if not success:
			break

		total_frames += 1
		num_frame += 1
		counter += 1
		# image = cv2.flip(image, 1)

		if num_frame >= interval:
			num_frame = 0
			if 'multipose' in estimation_model:
				# Run pose estimation using a MultiPose model.
				list_persons = pose_detector.detect(image)
			else:
				# Run pose estimation using a SinglePose model, and wrap the result in an
				# array.
				list_persons = [pose_detector.detect(image)]

			if len(list_persons)!=0:
				curr_time = time.time()
				pose_detector.update_data(list_persons, prev_data, image, curr_time)
				fall_detected, fall_conf = fall_detector.fall_detection_v2(prev_data, frame_width=vid_width, frame_height=vid_height)


		# write to frame if fall detected
		if fall_detected and fall_conf>=0.7:
			cv2.putText(image, "Fall Detected", (20,30),  cv2.FONT_HERSHEY_PLAIN,2,(0,0,245),3)
			
		# write person state (ie. sitting, standing, lying down) to frame
		for prev_data_key, prev_data_value in prev_data.copy().items():
			try:
				draw_point = prev_data[prev_data_key]['hips'][-1]
				temp_state = prev_data[prev_data_key]['state'][-1]
				temp_text = "ID " + str(prev_data_key) + ": " + temp_state
				cv2.putText(image, temp_text, (draw_point[0]+10, draw_point[1]+20),  cv2.FONT_HERSHEY_PLAIN,1.5,(155,200,0),2)
			except:
				pass
			
			# Draw keypoints and edges on input image
			# image = movenet_utils.visualize(image, list_persons)


		# Calculate the FPS
		# if counter % fps_avg_frame_count == 0:
		# 	end_time_fps = time.time()
		# 	fps_track = fps_avg_frame_count / (end_time_fps - start_time_fps)
		# 	start_time_fps = time.time()

		# Show the FPS
		fps_text = 'FPS = ' + str(int(fps_track))
		text_location = (left_margin, row_size)
		cv2.putText(image, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN,
					font_size, text_color, font_thickness)

		if save_video:
			video_output.write(image)
  
		# Stop the program if the ESC key is pressed.
		if cv2.waitKey(delay) == 27:
			break
		if to_show:
			cv2.imshow("Movenet Pose Fall Detection", image)
   
	# benchmark
	if benchmark:
		end_time = time.time()
		total_time = end_time - start_time
		total_fps = total_frames/(total_time)
		print(f'Total Frames: {total_frames}, Total Time: {total_time}, Total FPS: {total_fps}')
 	# cleanup
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