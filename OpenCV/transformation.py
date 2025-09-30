import cv2 as cv
import numpy as np

img=cv.imread('Resources/Photos/cat.jpg')
cv.imshow('Cat',img)

def translate(img,x,y):
    transMat=np.float32([[1,0,x],[0,1,y]])
    dimensions=(img.shape[1],img.shape[0])
    return cv.warpAffine(img, transMat,dimensions)

# -X --> left
# -Y --> Up
# X --> right
# Y --> down

translated=translate(img,100,100)
cv.imshow("Translated",translated)

def rotate(img,angle,rotPoint=None):
    (height,width)=img.shape[:2]
    if rotPoint is None:    #assuming roatating about center
        rotPoint=(width//2,height//2)
    rotMat=cv.getRotationMatrix2D(rotPoint,angle,1.0)
    dimensions=(width,height)

    return cv.warpAffine(img,rotMat,dimensions)

rotated=rotate(img,45) #positive value for anticlock wise
cv.imshow("Rotated",rotated)

#Flipping (0-> around x | 1-> around y | -1-> both)
flip=cv.flip(img,-1)
cv.imshow("Flipped",flip)

cv.waitKey(0)