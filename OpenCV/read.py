import cv2 as cv
#Read photo
img =cv.imread('Resources/Photos/cat.jpg')
cv.imshow('Cat',img)

#Reading video
capture=cv.VideoCapture('Resources/Videos/dog.mp4')

while True:
    isTrue, frame=capture.read()
    cv.imshow('Video',frame)
    if cv.waitKey(20) & 0xFF==ord('d'):
        break
capture.release()
cv.destroyAllWindows()