import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import math
import urllib.request
import argparse
import cv2
import time
import pathlib

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


def quantized_scale(name, state, input_details):
    """Scales the named state tensor input for the quantized model."""
    dtype = input_details[name]['dtype']
    scale, zero_point = input_details[name]['quantization']
    if 'frame_count' in name or dtype == np.float32 or scale == 0.0:
        return state
    return np.cast((state / scale + zero_point), dtype)

def get_top_k(probs, k, label_map):
    """Outputs the top k model labels and probabilities on the given video.

    Args:
    probs: probability tensor of shape (num_frames, num_classes) that represents
    the probability of each class on each frame.
    k: the number of top predictions to select.
    label_map: a list of labels to map logit indices to label strings.

    Returns:
    a tuple of the top-k labels and probabilities.
    """
    # Sort predictions to find top_k
    top_predictions = tf.argsort(probs, axis=-1, direction='DESCENDING')[:k]
    # collect the labels of top_k predictions
    top_labels = tf.gather(label_map, top_predictions, axis=-1)
    # decode lablels
    top_labels = [label.astype('U13') for label in top_labels.numpy()]
    # top_k probabilities of the predictions
    top_probs = tf.gather(probs, top_predictions, axis=-1).numpy()
    return tuple(zip(top_labels, top_probs))


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
    model_path = "../models/movinet-a0-int8.tflite"
    interpreter = tf.lite.Interpreter(model_path=model_path)
    runner = interpreter.get_signature_runner()
    input_details = runner.get_input_details()

    labels_path = tf.keras.utils.get_file(
        fname='labels.txt',
        origin='https://raw.githubusercontent.com/tensorflow/models/f8af2291cced43fc9f1d9b41ddbf772ae7b0d7d2/official/projects/movinet/files/kinetics_600_labels.txt'
    )
    labels_path = pathlib.Path(labels_path)

    lines = labels_path.read_text().splitlines()
    KINETICS_600_LABELS = np.array([line.strip() for line in lines])

    # Create the initial states, scale quantized.
    init_states = {
        name: quantized_scale(name, np.zeros(x['shape'], dtype=x['dtype']), input_details)
        for name, x in input_details.items()
        if name != 'image'
    }
    states = init_states

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
        
        input_frame = cv2.resize(frame, dsize=(172, 172))
        input_tensor = tf.expand_dims(input_frame, axis=0)
        input_tensor = tf.cast(input_tensor, dtype=tf.float32)
        # print(input_tensor.shape)
        outputs = runner(**states, image=input_tensor)
        # `logits` will output predictions on each frame.
        logits = outputs.pop('logits')
        probs = tf.nn.softmax(logits)
        top_k = get_top_k(probs=probs, k=5, label_map=KINETICS_600_LABELS)
        print()
        for label, prob in top_k:
            print(label, prob)

        # fps track
        currentTime = time.time()
        fps = 1/(currentTime-prevtime)
        prevtime = currentTime

        
        # display frame
        cv2.imshow("Image",frame)

        # press q to exit
        if cv2.waitKey(10) & 0xff == ord('q'):
            break 
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()