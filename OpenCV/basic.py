import cv2 as cv

img=cv.imread('Resources/Photos/cat.jpg')
cv.imshow('Cat',img)

#convert photo to gray
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('Gray',gray)

#blur
blur=cv.GaussianBlur(img,(3,3),cv.BORDER_DEFAULT)
cv.imshow('Blur',blur)

#Edge cascade
canny=cv.Canny(img,125,175)
cv.imshow("Edges",canny)

#Dilating
dilated=cv.dilate(canny,(7,7),iterations=3)
cv.imshow("Dilated",dilated)

#resize
resized=cv.resize(img,(500,500))
cv.imshow("Resized",resized)

#cropped
cropped=img[50:200,200:400]
cv.imshow("Cropped",cropped)

cv.waitKey(0)
