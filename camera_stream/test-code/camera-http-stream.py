import cv2
from flask import Flask, Response
import utils


app = Flask(__name__)

args = utils.get_args()

feed_src = args.src
port = args.port


print(args)

def generate_frames():
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
    
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=port)