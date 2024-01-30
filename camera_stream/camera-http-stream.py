""" 
This script creates an HTTP camera stream using Flask.
"""

import cv2
from flask import Flask, Response
import argparse

def get_args():
    """ Parse arguments. Returns user/default set arguments """
    parser = argparse.ArgumentParser(description="Pose estimate program")
    # user settings
    parser.add_argument("--src", type=str, default='/dev/video0', help="Video file location eg. /dev/video0")
    parser.add_argument("--show", type=int, default=1, help="Whether to show camera to screen, 0 to hide 1 to show.")
    parser.add_argument("--width", type=int, default=640, help="Input video width.")
    parser.add_argument("--height", type=int, default=480, help="Input video height")
    parser.add_argument("--port", type=int, default=5002, help="Port to run video feed")
    
    args = parser.parse_args()
    return args

def generate_frames():
    """ Create frames from the camera to stream. """
    camera = cv2.VideoCapture(feed_src)  # Replace with the appropriate camera index if you have multiple cameras

    while True:
        success, frame = camera.read()

        if not success:
            break

        # Convert the frame to JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield the frame as an HTTP response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    camera.release()

# create flask app
app = Flask(__name__)

# get user args
args = get_args()
feed_src = args.src
port = args.port

@app.route('/video_feed')
def video_feed():
    """ Return the camera feed as HTTP video stream."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    """ Run the flask app """
    app.run(host='0.0.0.0', port=port)