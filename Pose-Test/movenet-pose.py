import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
import time
import argparse
import os
import copy
from my_def import *

def get_args():
    """ Parse arguments """
    parser = argparse.ArgumentParser()
    # user settings
    parser.add_argument("--src", type=str, default='/dev/video0')
    parser.add_argument("--show", type=int, default=1)
    parser.add_argument("--width", help='cap width', type=int, default=640)
    parser.add_argument("--height", help='cap height', type=int, default=480)
    parser.add_argument("--conf_score", type=float, default=0.4)

    args = parser.parse_args()
    return args


def movenet(interpreter, input_image):
    """Runs detection on an input image.

    Args:
      input_image: A [1, height, width, 3] tensor represents the input image
        pixels. Note that the height/width should already be resized and match the
        expected input resolution of the model before passing into this function.

    Returns:
      A [1, 1, 17, 3] float numpy array representing the predicted keypoint
      coordinates and scores.
    """
    # TF Lite format expects tensor type of uint8.
    input_image = tf.cast(input_image, dtype=tf.uint8)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], input_image.numpy())
    # Invoke inference.
    interpreter.invoke()
    # Get the model prediction.
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
    return keypoints_with_scores

def keypoints_and_edges_for_display(keypoints_with_scores,
                                     height,
                                     width,
                                     keypoint_threshold=0.4):
  """Returns high confidence keypoints and edges for visualization.

  Args:
    keypoints_with_scores: A numpy array with shape [1, 1, 17, 3] representing
      the keypoint coordinates and scores returned from the MoveNet model.
    height: height of the image in pixels.
    width: width of the image in pixels.
    keypoint_threshold: minimum confidence score for a keypoint to be
      visualized.

  Returns:
    A (keypoints_xy, edges_xy, edge_colors) containing:
      * the coordinates of all keypoints of all detected entities;
      * the coordinates of all skeleton edges of all detected entities;
      * the colors in which the edges should be plotted.
  """
  keypoints_all = []
  keypoint_edges_all = []
  edge_colors = []
  num_instances, _, _, _ = keypoints_with_scores.shape
  for idx in range(num_instances):
    kpts_x = keypoints_with_scores[0, idx, :, 1]
    kpts_y = keypoints_with_scores[0, idx, :, 0]
    kpts_scores = keypoints_with_scores[0, idx, :, 2]
    kpts_absolute_xy = np.stack(
        [width * np.array(kpts_x), height * np.array(kpts_y)], axis=-1)
    kpts_above_thresh_absolute = kpts_absolute_xy[
        kpts_scores > keypoint_threshold, :]
    keypoints_all.append(kpts_above_thresh_absolute)

    for edge_pair, color in KEYPOINT_EDGE_INDS_TO_COLOR.items():
      if (kpts_scores[edge_pair[0]] > keypoint_threshold and
          kpts_scores[edge_pair[1]] > keypoint_threshold):
        x_start = kpts_absolute_xy[edge_pair[0], 0]
        y_start = kpts_absolute_xy[edge_pair[0], 1]
        x_end = kpts_absolute_xy[edge_pair[1], 0]
        y_end = kpts_absolute_xy[edge_pair[1], 1]
        line_seg = np.array([[x_start, y_start], [x_end, y_end]])
        keypoint_edges_all.append(line_seg)
        edge_colors.append(color)
  if keypoints_all:
    keypoints_xy = np.concatenate(keypoints_all, axis=0)
  else:
    keypoints_xy = np.zeros((0, 17, 2))

  if keypoint_edges_all:
    edges_xy = np.stack(keypoint_edges_all, axis=0)
  else:
    edges_xy = np.zeros((0, 2, 2))
  return keypoints_xy, edges_xy, edge_colors


def main():
    # parse arguments
    args = get_args()
    media_source = args.src
    show_cv = bool(args.show)
    cap_width = args.width
    cap_height = args.height

    # get and load model
    # model_url = "https://tfhub.dev/google/movenet/multipose/lightning/1"
    input_size = 192
    # Initialize the TFLite interpreter
    interpreter = tf.lite.Interpreter(model_path="../models/movenet-singlepose-lightning-f16.tflite")
    interpreter.allocate_tensors()



    # setup video capture
    cap = cv2.VideoCapture(media_source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)
    
    prevtime = 0
    
    while True:
        # read video frame
        ret, frame = cap.read()
        if not ret:
            break
        # Resize and pad the image to keep the aspect ratio and fit the expected size.
        input_image = tf.expand_dims(frame, axis=0)
        input_image = tf.image.resize_with_pad(input_image, input_size, input_size)

        # run inference
        keypoints_with_scores =  movenet(interpreter, input_image)

        keypoints_xy, edges_xy, edge_colors = keypoints_and_edges_for_display(keypoints_with_scores, frame.shape[1], frame.shape[0])
        print(f'{keypoints_xy}, {edges_xy}, {edge_colors}')
        # fps track
        currentTime = time.time()
        fps = 1/(currentTime-prevtime)
        prevtime = currentTime
        cv2.putText(frame, str(int(fps)), (50,50),  cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)

        # display frame
        cv2.imshow("Image",frame)

        # press q to exit
        if cv2.waitKey(10) & 0xff == ord('q'):
            break 
    
    cap.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()



