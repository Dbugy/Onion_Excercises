import numpy as np
import cv2


# Load an color image in grayscale
img = cv2.imread('onion.jpeg',0)



cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.waitKey(0)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('onion.png',img)
