# Edge Detection

import cv2 as cv
import numpy as np

img=cv.imread("Resources/Photos/park.jpg")
cv.imshow("Park",img)
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow("Gray",gray)

#Laplaction
lap=cv.Laplacian(gray,cv.CV_64F)
lap=np.uint8(np.absolute(lap))
cv.imshow("laplacian",lap)

#Sobel
sobelx=cv.Sobel(gray,cv.CV_64F,1,0)
sobely=cv.Sobel(gray,cv.CV_64F,0,1)
combined_sobel=cv.bitwise_or(sobelx,sobely)

cv.imshow("Sobel x",sobelx)
cv.imshow("Sobel Y",sobely)
cv.imshow("Sobel combined",combined_sobel)

#comparing both with canny
canny=cv.Canny(gray,150,175)
cv.imshow("Canny",canny)

cv.waitKey(0)