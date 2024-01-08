import tensorflow as tf
import numpy as np
import cv2
import time
import argparse
import os
import copy
import math
import utils
import my_defs


def detect_tflite(interpreter, input_tensor, image_width, image_height):
    """Runs detection on an input image.

    Args:
    interpreter: tf.lite.Interpreter
    input_tensor: A [1, input_height, input_width, 3] Tensor of type tf.float32.
        input_size is specified when converting the model to TFLite.

    Returns:
    A tensor of shape [1, 6, 56].
    """


    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    is_dynamic_shape_model = input_details[0]['shape_signature'][2] == -1
    if is_dynamic_shape_model:
        input_tensor_index = input_details[0]['index']
        input_shape = input_tensor.shape
        interpreter.resize_tensor_input(
            input_tensor_index, input_shape, strict=True)
    interpreter.allocate_tensors()

    interpreter.set_tensor(input_details[0]['index'], input_tensor.numpy())

    interpreter.invoke()

    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
    keypoints_with_scores = np.squeeze(keypoints_with_scores)
    #   return keypoints_with_scores
    keypoints_list, scores_list = [], []
    bbox_list = []
    for keypoints_with_score in keypoints_with_scores:
        keypoints = []
        scores = []
        # extract keypoints
        for index in range(17):
            keypoint_x = int(image_width *
                             keypoints_with_score[(index * 3) + 1])
            keypoint_y = int(image_height *
                             keypoints_with_score[(index * 3) + 0])
            score = keypoints_with_score[(index * 3) + 2]

            keypoints.append([keypoint_x, keypoint_y])
            scores.append(score)

        # boundary box points for each person
        bbox_ymin = int(image_height * keypoints_with_score[51])
        bbox_xmin = int(image_width * keypoints_with_score[52])
        bbox_ymax = int(image_height * keypoints_with_score[53])
        bbox_xmax = int(image_width * keypoints_with_score[54])
        bbox_score = keypoints_with_score[55]

        # list for storing at most 6 people
        keypoints_list.append(keypoints)
        scores_list.append(scores)
        bbox_list.append(
            [bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax, bbox_score])

    return keypoints_list, scores_list, bbox_list


def keep_aspect_ratio_resizer(image, target_size):
    """Resizes the image.

    The function resizes the image such that its longer side matches the required
    target_size while keeping the image aspect ratio. Note that the resizes image
    is padded such that both height and width are a multiple of 32, which is
    required by the model.
    """
    _, height, width, _ = image.shape
    if height > width:
        scale = float(target_size / height)
        target_height = target_size
        scaled_width = math.ceil(width * scale)
        image = tf.image.resize(image, [target_height, scaled_width])
        target_width = int(math.ceil(scaled_width / 32) * 32)
    else:
        scale = float(target_size / width)
        target_width = target_size
        scaled_height = math.ceil(height * scale)
        image = tf.image.resize(image, [scaled_height, target_width])
        target_height = int(math.ceil(scaled_height / 32) * 32)
    image = tf.image.pad_to_bounding_box(image, 0, 0, target_height, target_width)
    return (image,  (target_height, target_width))

def get_xy(scores, keypts, part,  conf_threshold):
    if scores[my_defs.KEYPOINT_DICT[part]] >  conf_threshold:
        return keypts[my_defs.KEYPOINT_DICT[part]]
        
    else:
        return (0,0)

def main():
    # parse arguments
    args = utils.get_args()
    media_source = args.src
    show_cv = bool(args.show)
    cap_width = args.width
    cap_height = args.height
    conf_threshold = args.conf_score

    # Initialize the TFLite interpreter
    # model_url = "https://tfhub.dev/google/movenet/multipose/lightning/1"
    input_size = 256
    model_path = "./models/movenet-multipose-lightning-f16.tflite"
    interpreter = tf.lite.Interpreter(model_path=model_path)
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
        
        image_width, image_height = frame.shape[1], frame.shape[0]
        
        # Resize and pad the image to keep the aspect ratio and fit the expected size.
        image = tf.expand_dims(frame, axis=0)
        resized_image, image_shape = keep_aspect_ratio_resizer(image, input_size)
        image_tensor = tf.cast(resized_image, dtype=tf.uint8)

        # run inference
        keypoints_list, scores_list, bbox_list =  detect_tflite(interpreter, tf.cast(image_tensor, dtype=tf.uint8), image_width, image_height)
        
        # fps track
        currentTime = time.time()
        fps = 1/(currentTime-prevtime)
        cv2.putText(frame, str(int(fps)), (50,50),  cv2.FONT_HERSHEY_PLAIN,3,(225,0,0),3)
        prevtime = currentTime

        # draw detections to frame
        i = 0
        for keypts, scores in zip(keypoints_list, scores_list):
            left_shoulder = get_xy(scores, keypts, 'left_shoulder', conf_threshold)
            right_shoulder = get_xy(scores, keypts, 'right_shoulder', conf_threshold)
            left_hip = get_xy(scores, keypts, 'left_hip', conf_threshold)
            right_hip = get_xy(scores, keypts, 'right_hip', conf_threshold)
            left_knee = get_xy(scores, keypts, 'left_knee', conf_threshold)
            right_knee = get_xy(scores, keypts, 'right_knee', conf_threshold)
            
            shoulder = utils.get_mainpoint(left_shoulder, right_shoulder, scores[my_defs.KEYPOINT_DICT['left_shoulder']], scores[my_defs.KEYPOINT_DICT['right_shoulder']], part="shoulder")
            hips = utils.get_mainpoint(left_hip, right_hip, scores[my_defs.KEYPOINT_DICT['left_hip']], scores[my_defs.KEYPOINT_DICT['right_hip']], part="hips")
            knees = utils.get_mainpoint(left_knee, right_knee, scores[my_defs.KEYPOINT_DICT['left_hip']], scores[my_defs.KEYPOINT_DICT['right_hip']], part="knees")
            shoulder_exist = False
            hips_exist = False
            knees_exist = False
            # if relevant keypt exist draw pt
            if shoulder!=(0,0):
                shoulder_exist = True
                utils.draw_keypoint(frame, shoulder)
                # print(f'Shoulder: {shoulder}')
            if hips!=(0,0):
                hips_exist = True
                utils.draw_keypoint(frame, hips)
                # print(f'Hips: {hips}')
            if knees!=(0,0):
                knees_exist = True
                utils.draw_keypoint(frame, knees)
                # print(f'Knees: {knees}')
                
             # if keypts exist draw line to connect them, calculate vector
            spine_vector = (0,0)
            legs_vector = (0,0)
            if shoulder_exist and hips_exist:
                spine_vector = utils.calculate_vector(hips, shoulder)
                # utils.draw_keypoint_line(frame, shoulder, hips)
                utils.draw_vector(frame, hips, spine_vector)
                # print(f'Spine Vector: {spine_vector}')
            if hips_exist and knees_exist:
                legs_vector = utils.calculate_vector(hips, knees)
                # utils.draw_keypoint_line(frame, hips, knees)
                utils.draw_vector(frame, hips, legs_vector)
                # print(f'Leg Vector: {legs_vector}')
            
            # calculate vector if all 3 main pts exist
            spine_leg_theta = -1 # angle between spine (vector between shoulder and hips) and legs (vector between hips and knees)
            spine_x_axis_phi = -1 # angle between spine (vector between shoulder and hips) and x_axis along hip point
            if shoulder_exist and hips_exist and knees_exist:
                spine_leg_theta = utils.angle_between(spine_vector, legs_vector)
                hips_x_axis = utils.calculate_vector(hips, (hips[0]+20, hips[1]))
                hips_y_axis = utils.calculate_vector(hips, (hips[0], hips[1]+20))
                # utils.draw_vector(frame, hips, hips_x_axis, color=(255,255,255))
                # spine_x_axis_phi = utils.calculate_angle_with_x_axis(spine_vector)
                spine_x_axis_phi = utils.angle_between(spine_vector, hips_x_axis)
                legs_y_axis_alpha = utils.angle_between(legs_vector, hips_y_axis)
                print(f'Person {i+1}')
                print(f'Theta: {spine_leg_theta}, Phi: {spine_x_axis_phi}, Alpha: {legs_y_axis_alpha}')
                state = utils.action_state(spine_leg_theta, spine_x_axis_phi, legs_y_axis_alpha)
                print(f'State: {state}')
                i += 1
            
        # display frame
        cv2.imshow("Image",frame)

        # press q to exit
        key = cv2.waitKey(1)
        if key == ord('q'):
            break 
    
    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()