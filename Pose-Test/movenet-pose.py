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


def run_inference_tflite(interpreter, input_size, image, use_tflite):
    input_image = tf.cast(image, dtype=tf.uint8)  
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], input_image.numpy())
    interpreter.invoke()

    outputs = interpreter.get_tensor(output_details[0]['index'])
    print(outputs)
    # 
    keypoints_list, scores_list = [], []
    bbox_list = []
    for keypoints_with_score in keypoints_with_scores:
        keypoints = []
        scores = []
        # 
        for index in range(17):
            keypoint_x = int(image_width *
                             keypoints_with_score[(index * 3) + 1])
            keypoint_y = int(image_height *
                             keypoints_with_score[(index * 3) + 0])
            score = keypoints_with_score[(index * 3) + 2]

            keypoints.append([keypoint_x, keypoint_y])
            scores.append(score)

        # 
        bbox_ymin = int(image_height * keypoints_with_score[51])
        bbox_xmin = int(image_width * keypoints_with_score[52])
        bbox_ymax = int(image_height * keypoints_with_score[53])
        bbox_xmax = int(image_width * keypoints_with_score[54])
        bbox_score = keypoints_with_score[55]

        # 6人分のデータ格納用のリストに追加
        keypoints_list.append(keypoints)
        scores_list.append(scores)
        bbox_list.append(
            [bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax, bbox_score])

    return keypoints_list, scores_list, bbox_list


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
        keypoints_list, scores_list, bbox_list =  run_inference_tflite(interpreter, input_size, input_image, use_tflite=True)
        print(keypoints_list)
        
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



