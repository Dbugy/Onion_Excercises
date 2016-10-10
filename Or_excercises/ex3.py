import numpy as np
import cv2

# Create a black image
img = np.zeros((512,512,3), np.uint8)

# Draw a diagonal blue line with thickness of 5 px
cv2.line(img,(0,0),(511,511),(255,0,0),5)

font=cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img, 'Left', (0,), font, 1, (0.0,255),3,cv2.CV_AA)
#img = cv2.line(img,(0,0),(511,511),(255,0,0),5)

#img = cv2.rectangle(img,(384,0),(510,128),(0,255,0),3)

cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()