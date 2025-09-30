import cv2 as cv
import matplotlib.pyplot as plt
img =cv.imread("Resources/Photos/lady.jpg")
#cv.imshow("Lady",img)

# plt.imshow(img)   //prepares the image
# plt.show()        // actually displays it

#BGR to gray
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow("Gray",gray)

#BGR to HSV
hsv=cv.cvtColor(img,cv.COLOR_BGR2HSV)
cv.imshow("HSV",hsv)

#BGR to RGB
rgb=cv.cvtColor(img,cv.COLOR_BGR2RGB)
cv.imshow("RGB",rgb)
plt.imshow(rgb)
plt.show() 

cv.waitKey(0)