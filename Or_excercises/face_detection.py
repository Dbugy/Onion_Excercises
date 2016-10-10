import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('/home/o/Desktop/Programming_Projects/opencv-2.4.13/data/haarcascades/haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('/home/o/Desktop/Programming_Projects/opencv-2.4.13/data/haarcascades/haarcascade_eye.xml')
profile_cascade = cv2.CascadeClassifier('/home/o/Desktop/Programming_Projects/opencv-2.4.13/data/haarcascades/haarcascade_profileface.xml')

cap = cv2.VideoCapture(0)
N=2
_, temp = cap.read()
rows, cols, _ = temp.shape
rows=rows/N
cols=cols/N

font=cv2.FONT_HERSHEY_SIMPLEX

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    img = cv2.flip(frame,1)

    smallImg=cv2.resize(img, (cols,rows))
#    print img.shape
    # print smallImg.shape
#    img = cv2.imread(frame)
    gray = cv2.cvtColor(smallImg, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray,1.3,5)
#    print faces
    for (x,y,w,h) in faces:
        # print "aaaa"
        cv2.rectangle(img,(x*N,y*N),((x+w)*N,(y+h)*N),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y*N:(y+h)*N, x*N:(x+w)*N]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex*N,ey*N),((ex+ew)*N,(ey+eh)*N),(255,255,255),2)

    lProfile= profile_cascade.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in lProfile:
        cv2.rectangle(img, (x * N, y * N), ((x + w) * N, (y + h) * N), (0, 255, 0), 2)
        cv2.putText(img, 'Left', (cols*N/2,rows*N-5), font, 1, (0,255,0),3,cv2.CV_AA)

    rProfile = profile_cascade.detectMultiScale(cv2.flip(gray,1), 1.3, 5)
    for (x, y, w, h) in rProfile:
        cv2.rectangle(img, ((cols-x-w)*N, y * N), ((cols-x) * N, (y + h) * N), (0, 0, 255), 2)
        cv2.putText(img, 'Right', (cols * N / 2, rows * N - 5), font, 1, (0, 0, 255), 3, cv2.CV_AA)

    cv2.imshow('img', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#cv2.imshow('img',img)
#cv2.waitKey(0)
cv2.destroyAllWindows()