import os 
import cv2 as cv
import numpy as np

#manually enter the name of faces
people=['Ben Affleck','Elton John','Jerry Seinfield','Madonna','Mindy Kaling']



DIR=r'C:\Users\kshit\OneDrive\Desktop\lane\OpenCV\Resources\Faces'
haar_cascade=cv.CascadeClassifier('haar_face.xml')

features=[] #will store cropped-out face images (face regions).
labels=[] #will store the index number of the person (like 0 for Ben Afflek, 1 for Elton John, etc.).

def create_train():
    for person in people:
        path=os.path.join(DIR,person) #Builds the full path to that person’s folder
        label=people.index(person) #label assigns a number to the person

        for img in os.listdir(path): # img in list of all entries of path ie. person's folder
            img_path=os.path.join(path,img) #going further in the path for particular image

            img_array=cv.imread(img_path)
            if img_array is None: #if image can't be read skip it
                continue
            
            gray=cv.cvtColor(img_array,cv.COLOR_BGR2GRAY)
            faces_rect=haar_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=4)
            #returned list of rectangles (x,y,w,h) of detected face

            #For each detected face:
            for(x,y,w,h)in faces_rect:
                faces_region_of_interst=gray[y:y+h,x:x+w] #crop out the face
                features.append(faces_region_of_interst)
                labels.append(label)

create_train()
print("Training done ---------")

features=np.array(features,dtype='object')
#Don’t try to force them into a rectangular block. Just store each ROI (NumPy array) as an object in a NumPy array
#So you end up with a 1D NumPy array of objects (each object = one face array)
labels=np.array(labels)
#makes a NumPy array of corresponding labels.


face_recognizer = cv.face.LBPHFaceRecognizer_create()
#This creates a Local Binary Patterns Histograms (LBPH) face recognizer.
#LBPH is a simple but effective algorithm that analyzes the texture of a face.


# Train the Recognizer on the features list and the labels list
face_recognizer.train(features,labels)

face_recognizer.save('face_trained.yml') #saving it for future use
np.save('features.npy', features)
np.save('labels.npy', labels)