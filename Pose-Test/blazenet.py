from mediapipe_pose_test import PoseModule
import cv2
import time
import mediapipe as mp
import argparse


if __name__ == "__main__":
    
    par = argparse.ArgumentParser(description='Human Fall Detection Demo.')
    par.add_argument('-v', '--video', default="/dev/video0",  # required=True,  # default=2,
                        help='Source of camera or video file path.')
    args = par.parse_args()
    print(args)
    vid_source = args.video
    print(vid_source)
    
    
    pose_obj = PoseModule()
    
    cap = cv2.VideoCapture(vid_source)
    prevtime = 0
    mp_pose = mp.solutions.pose

    while True:
        # read video frame
        ret, frame = cap.read()
        
        # test image intead of video
        # frame = cv2.imread(img_test)
        # frame = cv2.resize(frame, (320, 320))
        
        # annotate pose landmarks to iamge, return landmark results
        frame, results = pose_obj.find_pose(frame)
        
        # extract relevant landmarks for detection
        visibility_threshold = 0.7
        if results.pose_landmarks:
             # neck
            right_shoulder = results.pose_landmarks.landmark[14]
            left_shoulder = results.pose_landmarks.landmark[11]
            if right_shoulder.visibility>visibility_threshold and left_shoulder.visibility>visibility_threshold:
                shoulder = right_shoulder
            elif right_shoulder.visibility>visibility_threshold:
                shoulder = right_shoulder
            elif left_shoulder.visibility>visibility_threshold:
                shoulder = left_shoulder
            else:
                shoulder = -1
                
            # hip
            right_shoulder = results.pose_landmarks.landmark[14]
            left_shoulder = results.pose_landmarks.landmark[11]
            if right_shoulder.visibility>visibility_threshold and left_shoulder.visibility>visibility_threshold:
                shoulder = right_shoulder
            elif right_shoulder.visibility>visibility_threshold:
                shoulder = right_shoulder
            elif left_shoulder.visibility>visibility_threshold:
                shoulder = left_shoulder
            else:
                shoulder = -1
            
            # knee
            right_shoulder = results.pose_landmarks.landmark[14]
            left_shoulder = results.pose_landmarks.landmark[11]
            if right_shoulder.visibility>visibility_threshold and left_shoulder.visibility>visibility_threshold:
                shoulder = right_shoulder
            elif right_shoulder.visibility>visibility_threshold:
                shoulder = right_shoulder
            elif left_shoulder.visibility>visibility_threshold:
                shoulder = left_shoulder
            else:
                shoulder = -1
            
            if shoulder!=-1:
                print(shoulder)
            
        
            

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
    
    