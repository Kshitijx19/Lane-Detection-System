import cv2 as cv
img=cv.imread('road.jpg')
cv.imshow("Road",img)

gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow("Gray",gray)

blur=cv.GaussianBlur(gray,(5,5),cv.BORDER_DEFAULT)
cv.imshow("Blur",blur)

canny=cv.Canny(blur,125,175)
cv.imshow("Edges",canny)

cv.waitKey(0)