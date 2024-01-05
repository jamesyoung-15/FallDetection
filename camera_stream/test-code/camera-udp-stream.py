import cv2
import socket

def start_camera_stream():
    # Create a UDP socket
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # IP address and port to send the frames
    target_ip = '127.0.0.1'  # Replace with the IP address of the receiver
    target_port = 5555  # Replace with the desired port number

    # Open the USB webcam
    cap = cv2.VideoCapture(0)

    while True:
        # Read frame from the webcam
        ret, frame = cap.read()

        # Convert the frame to JPEG format
        ret, jpeg = cv2.imencode('.jpg', frame)

        # Send the frame as a UDP packet
        udp_socket.sendto(jpeg.tobytes(), (target_ip, target_port))

if __name__ == '__main__':
    start_camera_stream()