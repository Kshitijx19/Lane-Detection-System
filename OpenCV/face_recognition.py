import cv2 as cv
import numpy as np

haar_cascade=cv.CascadeClassifier('haar_face.xml')
people=['Ben Affleck','Elton John','Jerry Seinfield','Madonna','Mindy Kaling']

# features=np.load('features.npy')
# labels=np.load('labels.npy')

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

img=cv.imread(
    r'C:\Users\kshit\OneDrive\Desktop\lane\OpenCV\Resources\Madonna_test.webp'
)

gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow("person",gray)

#detect the person in the image
face_rect=haar_cascade.detectMultiScale(gray,1.1,4)
for(x,y,w,h) in face_rect:
    faces_roi=gray[y:y+h,x:x+w]
    label,confidence=face_recognizer.predict(faces_roi)
    print(f'Label ={people[label]} with a confidence of {confidence}')

    cv.putText(img,str(people[label]),(20,25),cv.FONT_HERSHEY_COMPLEX,1.0,(0,0,255),2)
    cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
cv.imshow("Detected face",img)
cv.waitKey(0)