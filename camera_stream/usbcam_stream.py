# import the necessary packages
from threading import Thread
import cv2


class USBCamStream:
	def __init__(self, src='/dev/video0'):
		""" initialize the video camera stream and read the first frame from the stream """
		self.stream = cv2.VideoCapture(src)
		(self.grabbed, self.frame) = self.stream.read()

	def start(self):
		""" start the thread to read frames from the video stream """
		self.thread = Thread(target=self.update, args=()).start()
		return self
	
	def resize_stream(self, width, height):
		""" Resize usb camera stream to specified width and height. """
		self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
		self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
	
	def read(self):
		""" Return the frame most recently read and grabbed status. """
		return self.grabbed, self.frame

	def release(self):
		""" Release the video stream. """
		self.stream.release()
  
	def change_format(self):
		""" Change the video stream format from YUYV to MJPG. """
		self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))