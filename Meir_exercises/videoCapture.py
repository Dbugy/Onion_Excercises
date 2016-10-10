import numpy as np
import cv2

#the 0 sets the cv2.videoCapture to use a webcame 
#else it would use a video

cap = cv2.VideoCapture(0)

#load cascade clasifiers

face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
side = cv2.CascadeClassifier('haarcascade_profileface.xml')
eye = cv2.CascadeClassifier('haarcascade_eye.xml')

N=2

while(True):

        # Capture frame-by-frame, ret is boolian, false if frame
        # not read correctly
        ret, frame = cap.read()

        rows, cols, _ = frame.shape

        gray = cv2.cvtColor(cv2.resize(frame,(cols/N,rows/N)), cv2.COLOR_BGR2GRAY)

        # Display the resulting frame

        faces = face.detectMultiScale(gray, 2, 5)

        if len(faces):
            for (x, _, _, _) in faces:
                x_co=x
            c1 = 255
            c2 = 0
            c3 = 0
        else :
            faces = side.detectMultiScale(gray, 2, 5)

            if len(faces):
                for (x, _, _, _) in faces:
                    x_co=x
                c1=0
                c2=255
                c3=0
            else:
                faces = side.detectMultiScale(cv2.flip(gray,1), 2, 5)
                for (x, _, w, _) in faces:
                    x_co=(cols/N)-x-w
                c1=0
                c2=0
                c3=255


        for (x,y,w,h) in faces:
            x_co *= N
            y *= N
            w *= N
            h *= N
            cv2.rectangle(frame,(x_co,y),(x_co+w,y+h),(c1,c2,c3),5)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            roi_gray = gray[y:y+h, x_co:x_co+w]
            roi_color = frame[y:y+h, x_co:x_co+w]
            eyes = eye.detectMultiScale(roi_gray, 1.3, 5)
            for (x,y,w,h) in eyes:
                cv2.circle(roi_color,(x+(w/2),y+(h/2)),(w/4) ,(0,0,0),w/2)
                cv2.circle(roi_color, (x + (w / 2), y + (h / 2)), (w/10), (255, 255, 255), w*2/3)
                cv2.circle(roi_color, (x + (w / 2), y + (h / 2)), (w/10), (255, 0, 0), int(w/4))
                cv2.circle(roi_color, (x + (w / 2), y + (h / 2)), (w / 20), (0, 0, 0), int(w / 10))

        cv2.imshow('frame',frame)
        #cv2.imshow('frame',cv2.flip(gray,1))
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

