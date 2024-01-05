import cv2
from collections import deque

# Initialize the video capture
camera = cv2.VideoCapture(2)

# Define the size of the frame buffer
buffer_size = 3

# Create a buffer using a list
frame_buffer = []

while True:
    # Read a frame from the camera
    ret, frame = camera.read()

    # Add the frame to the buffer
    frame_buffer.append(frame)

    # Check if the buffer size exceeds the desired size
    if len(frame_buffer) > buffer_size:
        # Remove the oldest frame from the buffer
        frame_buffer.pop(0)
    
    # Perform your desired actions using the frames in the buffer
    i = 0
    for single_frame in frame_buffer:
        # Process each frame in the buffer
        # Example: Perform object detection, apply filters, etc.
        cv2.putText(single_frame, str(int(i)), (100,100),  cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)
        # Display the frame
        cv2.imshow('Video', single_frame)
        
    # Check for the 'q' key to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
camera.release()
cv2.destroyAllWindows()