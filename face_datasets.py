import cv2
import numpy as np


detector=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#Using Classifier for face detecting
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
#Using Classifier for eye_detecting
cam = cv2.VideoCapture(0)
#VideoCapture

Id=raw_input('Enter Your Name::')#Console Input
Count=0#Count Value For counting and stroing images

while(True):
    ret, img = cam.read()#Read image and return two values
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#Converting the color to gray
    faces = detector.detectMultiScale(gray, 1.3, 5)#Detecting the face
    
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)#Drawing a rectangle for the image
        roi_gray = gray[y:y+h, x:x+w]#Fetching face from gray 
        roi_eyes = img[y:y+h, x:x+w]#Fetching eyes from image
        
        #incrementing sample Count 
        Count=Count+1
        #saving the captured face in the dataset folder
        cv2.imwrite("dataSet/User."+Id +'.'+ str(Count) + ".jpg", gray[y:y+h,x:x+w])

          
        eyes = eye_cascade.detectMultiScale(roi_gray)#detecting eyes
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_eyes,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)#drawing eyes for rectangle

        cv2.imshow('frame',img)
    #wait for 100 miliseconds 
    if cv2.waitKey(100) & 0xFF == ord('a'):
        break
    # break if the sample Count is morethan 20
    elif Count>20:
        break
cam.release()
cv2.destroyAllWindows()
