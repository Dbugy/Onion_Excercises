import cv2
import sys

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
profileCascade = cv2.CascadeClassifier('haarcascade_profileface.xml')

video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, mframe = video_capture.read()
    mframe = cv2.resize(mframe, (600,300))
    frame = cv2.resize(mframe, (100,50))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rightprofiles = profileCascade.detectMultiScale(
    	gray,
	scaleFactor=1.1,
	minNeighbors=2,
	minSize=(10,10),
	flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    flip_frame = cv2.flip(frame,1)
    gray_flip = cv2.flip(gray,1)

    leftprofiles = profileCascade.detectMultiScale(
	gray_flip,
	scaleFactor=1.1,
	minNeighbors=2,
	minSize=(10,10),
	flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=2,
        minSize=(10, 10),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    mframe_flipped = cv2.flip(mframe,1)
    # Draw a rectangle around the faces
    for (x, y, w, h) in leftprofiles:
	cv2.rectangle(mframe_flipped, (x*6,y*6),((x+w)*6,(y+h)*6), (255,0,0),2)

    mframe = cv2.flip(mframe_flipped,1)

    for (x, y, w, h) in faces:
        cv2.rectangle(mframe, (x*6, y*6), ((x+w)*6, (y+h)*6), (0, 0, 255), 2)

    for (x, y, w, h) in rightprofiles:
	cv2.rectangle(mframe, (x*6,y*6),((x+w)*6, (y+h)*6), (0,255,0),2)

    mframe = cv2.flip(mframe, 1)
    # Display the resulting frame
    cv2.imshow('Video', mframe)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
