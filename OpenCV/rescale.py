import cv2 as cv

def rescale_frame(frame,scale=0.35):
    width=int(frame.shape[1]*scale)
    height=int(frame.shape[0]*scale)
    dimensions=(width,height)
    return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)

#reading video 
capture=cv.VideoCapture('Resources/Videos/dog.mp4')
while True:
    isTrue, frame =capture.read()
    
    #rescaling video
    frame_resized=rescale_frame(frame)

    cv.imshow('Video',frame)
    cv.imshow('Video resized',frame_resized)
    if cv.waitKey(20) & 0xFF==ord('d'):
        break
capture.release()
cv.destroyAllWindows()

