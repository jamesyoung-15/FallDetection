# import the necessary packages
from camera_stream.usbcam_stream import USBCamStream
import cv2
import utils
import time

args = utils.get_args()
print(args)
vid_src = args.src
stream_width = args.width
stream_height = args.height


# setup threaded video stream
stream = USBCamStream(src=vid_src)
stream.resize_stream(stream_width,stream_height)
stream = stream.start()
# stream.set_fps(5)
prev_time = 0

# loop over some frames
while True:
    # get frame
    grabbed, frame = stream.read()
    
    # track fps
    current_time = time.time()
    fps = 1/(current_time-prev_time)
    prev_time = current_time
    cv2.putText(frame, str(int(fps)), (50,50),  cv2.FONT_HERSHEY_PLAIN,3,(225,0,0),3)
    
    # show the frame
    cv2.imshow("Frame", frame)
    
    # press q to quit
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# cleanup
stream.stop()
stream.release()
cv2.destroyAllWindows()