import cv2 as cv
import numpy as np

blank=np.zeros((500,500,3),dtype='uint8')
cv.imshow('Blank',blank)

# blank[200:300,300:400]=255,0,0
# cv.imshow('Blue',blank)

cv.rectangle(blank,(10,10),(blank.shape[1]//2,blank.shape[0]//2),(0,255,0),thickness=-1)
cv.circle(blank,(blank.shape[1]//2,blank.shape[0]//2),50,(0,0,255),thickness=-1)
cv.line(blank,(0,0),(250,250),(255,255,255),thickness=5)
# cv.imshow("line",blank)
# cv.imshow("circle",blank)

#put text on the image
cv.putText(blank,'Hello',(400,400),cv.FONT_HERSHEY_TRIPLEX,1.0,(255,255,0),thickness=2)
cv.imshow("rectangle",blank)
cv.waitKey(0)