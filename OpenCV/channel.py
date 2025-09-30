import cv2 as cv
import numpy as np

img =cv.imread("Resources/Photos/park.jpg")
cv.imshow("Park",img)

#creating blank
blank=np.zeros(img.shape[:2],dtype='uint8')

#spliting to B G R
b,g,r=cv.split(img)
cv.imshow("Blue",b)
cv.imshow("Green",g)
cv.imshow("Red",r)
print(img.shape)
print(b.shape)
print(g.shape)
print(r.shape)

#showing only particular color on img
blue=cv.merge([b,blank,blank])
green=cv.merge([blank,g,blank])
red=cv.merge([blank,blank,r])
cv.imshow("only_Blue",blue)
cv.imshow("only_Green",green)
cv.imshow("only_Red",red)

#merge them
merge=cv.merge([b,g,r])
cv.imshow("Merged",merge)

cv.waitKey(0)