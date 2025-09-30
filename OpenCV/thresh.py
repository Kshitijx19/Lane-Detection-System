import cv2 as cv
img =cv.imread('Resources/Photos/cats.jpg')
cv.imshow('Cats',img)

gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow("Gray",gray)

#Thresholding is a way to convert a grayscale image into a binary image.
#Each pixel is compared to a threshold value and replaced accordingly.
# here --> Pixels above 150 → white (255), below → black (0)


#simple Threshold
threshold, thresh =cv.threshold(gray,150,255,cv.THRESH_BINARY)
cv.imshow("Simple Threshold",thresh)
threshold, thresh_inv =cv.threshold(gray,150,255,cv.THRESH_BINARY_INV)
cv.imshow("Simple Threshold Inverse",thresh_inv)


#Adaptive Threshold
adaptive_thresh=cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,11,3)
cv.imshow("Adaptive Thresholding",adaptive_thresh)

cv.waitKey(0)