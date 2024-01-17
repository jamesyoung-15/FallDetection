from ultralytics import YOLO
import argparse
import cv2

par = argparse.ArgumentParser(description='Human Fall Detection Demo.')
par.add_argument('-src', '--src', default=0,  help='Source of camera or video file path. Eg. /dev/video0 or ./videos/myvideo.mp4')
# par.add_argument('-s', '--stream', default=1, help='Specify whether stream or not. 0 for true 1 for false.')
# par.add_argument('-o', '--show', default=1, help='Specify whether to show output or not. 0 for true 1 for false.')
args = par.parse_args()
print(args)
vid_source = args.src


model = YOLO('../yolo-weights/yolov8n-pose.pt')

# Open the video file
cap = cv2.VideoCapture(vid_source)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()