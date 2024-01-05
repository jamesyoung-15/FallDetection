# import the necessary packages
from camera_stream.usbcam_stream import USBCamStream
import cv2
import utils
import time

args = utils.get_args()
print(args)
vid_src = args.src

# grab a pointer to the video stream and initialize the FPS counter
print("[INFO] sampling frames from webcam...")
stream = USBCamStream(src=vid_src).start()
prev_time = 0
# loop over some frames
while True:
	# grab the frame from the stream and resize it to have a maximum
	# width of 400 pixels
    grabbed, frame = stream.read()
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    # update the FPS counter
    current_time = time.time()
    fps = 1/(current_time - prev_time)
    cv2.putText(frame, str(int(fps)), (50,50),  cv2.FONT_HERSHEY_PLAIN,3,(225,0,0),3)
    prev_time = current_time
    
    if key == ord('q'):
        break

# do a bit of cleanup
stream.release()
cv2.destroyAllWindows()