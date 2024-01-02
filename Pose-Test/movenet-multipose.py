import tensorflow as tf
import numpy as np
import cv2
import time
import argparse
import os
import copy
import math


def get_args():
    """ Parse arguments. Returns user/default set arguments """
    parser = argparse.ArgumentParser()
    # user settings
    parser.add_argument("--src", type=str, default='/dev/video0')
    parser.add_argument("--show", type=int, default=1)
    parser.add_argument("--width", help='cap width', type=int, default=640)
    parser.add_argument("--height", help='cap height', type=int, default=480)
    parser.add_argument("--conf_score", type=float, default=0.4)

    args = parser.parse_args()
    return args


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


def main():
    # parse arguments
    args = get_args()
    media_source = args.src
    show_cv = bool(args.show)
    cap_width = args.width
    cap_height = args.height

    # Initialize the TFLite interpreter
    # model_url = "https://tfhub.dev/google/movenet/multipose/lightning/1"
    input_size = 256
    model_path = "../models/movenet-multipose-lightning-f16.tflite"
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
        prevtime = currentTime

        # draw detections to frame
        frame = draw_detections(frame, 0.4, keypoints_list, scores_list, 0.2, bbox_list, fps)
        
        # display frame
        cv2.imshow("Image",frame)

        # press q to exit
        if cv2.waitKey(10) & 0xff == ord('q'):
            break 
    
    cap.release()
    cv2.destroyAllWindows()


def draw_detections(image, keypoint_score_th, keypoints_list, scores_list, bbox_score_th, bbox_list, fps):
    """ 
    Draws pose detections and boundary boxes around people.
    
    
    """
    display_image = copy.deepcopy(image)

    # 0:nose 1:left eye 2:右目 3:左耳 4:右耳 5:左肩 6:右肩 7:左肘 8:右肘 # 9:左手首
    # 10:右手首 11:左股関節 12:右股関節 13:左ひざ 14:右ひざ 15:左足首 16:右足首
    for keypoints, scores in zip(keypoints_list, scores_list):
        # Line：鼻 → 左目
        index01, index02 = 0, 1
        if scores[index01] > keypoint_score_th and scores[
                index02] > keypoint_score_th:
            point01 = keypoints[index01]
            point02 = keypoints[index02]
            cv2.line(display_image, point01, point02, (255, 255, 255), 4)
            cv2.line(display_image, point01, point02, (0, 0, 0), 2)
        # Line：鼻 → 右目
        index01, index02 = 0, 2
        if scores[index01] > keypoint_score_th and scores[
                index02] > keypoint_score_th:
            point01 = keypoints[index01]
            point02 = keypoints[index02]
            cv2.line(display_image, point01, point02, (255, 255, 255), 4)
            cv2.line(display_image, point01, point02, (0, 0, 0), 2)
        # Line：左目 → 左耳
        index01, index02 = 1, 3
        if scores[index01] > keypoint_score_th and scores[
                index02] > keypoint_score_th:
            point01 = keypoints[index01]
            point02 = keypoints[index02]
            cv2.line(display_image, point01, point02, (255, 255, 255), 4)
            cv2.line(display_image, point01, point02, (0, 0, 0), 2)
        # Line：右目 → 右耳
        index01, index02 = 2, 4
        if scores[index01] > keypoint_score_th and scores[
                index02] > keypoint_score_th:
            point01 = keypoints[index01]
            point02 = keypoints[index02]
            cv2.line(display_image, point01, point02, (255, 255, 255), 4)
            cv2.line(display_image, point01, point02, (0, 0, 0), 2)
        # Line：鼻 → 左肩
        index01, index02 = 0, 5
        if scores[index01] > keypoint_score_th and scores[
                index02] > keypoint_score_th:
            point01 = keypoints[index01]
            point02 = keypoints[index02]
            cv2.line(display_image, point01, point02, (255, 255, 255), 4)
            cv2.line(display_image, point01, point02, (0, 0, 0), 2)
        # Line：鼻 → 右肩
        index01, index02 = 0, 6
        if scores[index01] > keypoint_score_th and scores[
                index02] > keypoint_score_th:
            point01 = keypoints[index01]
            point02 = keypoints[index02]
            cv2.line(display_image, point01, point02, (255, 255, 255), 4)
            cv2.line(display_image, point01, point02, (0, 0, 0), 2)
        # Line：左肩 → 右肩
        index01, index02 = 5, 6
        if scores[index01] > keypoint_score_th and scores[
                index02] > keypoint_score_th:
            point01 = keypoints[index01]
            point02 = keypoints[index02]
            cv2.line(display_image, point01, point02, (255, 255, 255), 4)
            cv2.line(display_image, point01, point02, (0, 0, 0), 2)
        # Line：左肩 → 左肘
        index01, index02 = 5, 7
        if scores[index01] > keypoint_score_th and scores[
                index02] > keypoint_score_th:
            point01 = keypoints[index01]
            point02 = keypoints[index02]
            cv2.line(display_image, point01, point02, (255, 255, 255), 4)
            cv2.line(display_image, point01, point02, (0, 0, 0), 2)
        # Line：左肘 → 左手首
        index01, index02 = 7, 9
        if scores[index01] > keypoint_score_th and scores[
                index02] > keypoint_score_th:
            point01 = keypoints[index01]
            point02 = keypoints[index02]
            cv2.line(display_image, point01, point02, (255, 255, 255), 4)
            cv2.line(display_image, point01, point02, (0, 0, 0), 2)
        # Line：右肩 → 右肘
        index01, index02 = 6, 8
        if scores[index01] > keypoint_score_th and scores[
                index02] > keypoint_score_th:
            point01 = keypoints[index01]
            point02 = keypoints[index02]
            cv2.line(display_image, point01, point02, (255, 255, 255), 4)
            cv2.line(display_image, point01, point02, (0, 0, 0), 2)
        # Line：右肘 → 右手首
        index01, index02 = 8, 10
        if scores[index01] > keypoint_score_th and scores[
                index02] > keypoint_score_th:
            point01 = keypoints[index01]
            point02 = keypoints[index02]
            cv2.line(display_image, point01, point02, (255, 255, 255), 4)
            cv2.line(display_image, point01, point02, (0, 0, 0), 2)
        # Line：左股関節 → 右股関節
        index01, index02 = 11, 12
        if scores[index01] > keypoint_score_th and scores[
                index02] > keypoint_score_th:
            point01 = keypoints[index01]
            point02 = keypoints[index02]
            cv2.line(display_image, point01, point02, (255, 255, 255), 4)
            cv2.line(display_image, point01, point02, (0, 0, 0), 2)
        # Line：左肩 → 左股関節
        index01, index02 = 5, 11
        if scores[index01] > keypoint_score_th and scores[
                index02] > keypoint_score_th:
            point01 = keypoints[index01]
            point02 = keypoints[index02]
            cv2.line(display_image, point01, point02, (255, 255, 255), 4)
            cv2.line(display_image, point01, point02, (0, 0, 0), 2)
        # Line：左股関節 → 左ひざ
        index01, index02 = 11, 13
        if scores[index01] > keypoint_score_th and scores[
                index02] > keypoint_score_th:
            point01 = keypoints[index01]
            point02 = keypoints[index02]
            cv2.line(display_image, point01, point02, (255, 255, 255), 4)
            cv2.line(display_image, point01, point02, (0, 0, 0), 2)
        # Line：左ひざ → 左足首
        index01, index02 = 13, 15
        if scores[index01] > keypoint_score_th and scores[
                index02] > keypoint_score_th:
            point01 = keypoints[index01]
            point02 = keypoints[index02]
            cv2.line(display_image, point01, point02, (255, 255, 255), 4)
            cv2.line(display_image, point01, point02, (0, 0, 0), 2)
        # Line：右肩 → 右股関節
        index01, index02 = 6, 12
        if scores[index01] > keypoint_score_th and scores[
                index02] > keypoint_score_th:
            point01 = keypoints[index01]
            point02 = keypoints[index02]
            cv2.line(display_image, point01, point02, (255, 255, 255), 4)
            cv2.line(display_image, point01, point02, (0, 0, 0), 2)
        # Line：右股関節 → 右ひざ
        index01, index02 = 12, 14
        if scores[index01] > keypoint_score_th and scores[
                index02] > keypoint_score_th:
            point01 = keypoints[index01]
            point02 = keypoints[index02]
            cv2.line(display_image, point01, point02, (255, 255, 255), 4)
            cv2.line(display_image, point01, point02, (0, 0, 0), 2)
        # draw line from right knee to right ankle
        index01, index02 = 14, 16
        if scores[index01] > keypoint_score_th and scores[
                index02] > keypoint_score_th:
            point01 = keypoints[index01]
            point02 = keypoints[index02]
            cv2.line(display_image, point01, point02, (255, 255, 255), 4)
            cv2.line(display_image, point01, point02, (0, 0, 0), 2)

        # draw circle around keypoint
        for keypoint, score in zip(keypoints, scores):
            if score > keypoint_score_th:
                cv2.circle(display_image, keypoint, 6, (255, 255, 255), -1)
                cv2.circle(display_image, keypoint, 3, (0, 0, 0), -1)

    # draw bounding box around person
    for bbox in bbox_list:
        if bbox[4] > bbox_score_th:
            cv2.rectangle(display_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                         (255, 255, 255), 4)
            cv2.rectangle(display_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                         (0, 0, 0), 2)

    # draw fps to window
    cv2.putText(display_image, str(int(fps)), (50,50),  cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)

    return display_image

if __name__ == '__main__':
    main()



