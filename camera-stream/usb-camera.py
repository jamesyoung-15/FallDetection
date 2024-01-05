import cv2
import time
import utils

args = utils.get_args()

cap = cv2.VideoCapture(args.src)
cap.set(3, 640)
cap.set(4, 480)

prevtime = 0

while True:
    ret, img= cap.read()
    
     #fps track
    currentTime = time.time()
    fps = 1/(currentTime-prevtime)
    cv2.putText(img, str(int(fps)), (50,50),  cv2.FONT_HERSHEY_PLAIN,3,(225,0,0),3)
    prevtime = currentTime
    
    cv2.imshow('Webcam', img)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()