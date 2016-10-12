import cv2
import sys

count = 0
video_capture = cv2.VideoCapture(0)

while True:
	ret, frame = video_capture.read()
	
	frame = cv2.flip(frame, 1)
	cv2.imshow('Video', frame)
	count = count+1
	cv2.imwrite(str(count)+".jpg", frame)
	if cv2.waitKey(1) & 0xff == ord('q'):
		break

video_capture.release()
cv2.destroyAllWindows()

