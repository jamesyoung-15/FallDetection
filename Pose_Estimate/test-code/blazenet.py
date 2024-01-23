from mediapipe_pose import PoseModule
import cv2
import time
import mediapipe as mp
import argparse


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Human Fall Detection Demo.')
    parser.add_argument('-v', '--video', default="/dev/video0",  # required=True,  # default=2,
                        help='Source of camera or video file path.')
    parser.add_argument("--width", help='cap width', type=int, default=640)
    parser.add_argument("--height", help='cap height', type=int, default=480)
    args = parser.parse_args()
    vid_source = args.video
    cap_width = args.width
    cap_height = args.height
    
    
    pose_obj = PoseModule()
    
    cap = cv2.VideoCapture(vid_source)
    prevtime = 0
    mp_pose = mp.solutions.pose
    currFrame = 1
    while True:
        # read video frame
        ret, frame = cap.read()
        if ret == False:
            break
        currFrame += 1
        
        # test image intead of video
        # frame = cv2.imread(img_test)
        frame = cv2.resize(frame, (384, 384))
        if currFrame == 5:
            currFrame = 1
            # annotate pose landmarks to iamge, return landmark results
            frame, results = pose_obj.find_pose(frame)
        
            
        
            

        # fps track
        currentTime = time.time()
        fps = 1/(currentTime-prevtime)
        prevtime = currentTime
        # print fps
        cv2.putText(frame, str(int(fps)), (50,50),  cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)
        
        # display pose detections
        cv2.imshow("Image",frame)
        
        if cv2.waitKey(10) & 0xff == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
    