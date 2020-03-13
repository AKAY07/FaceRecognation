import cv2
import numpy as np

face_classifier=cv2.CascadeClassifier('C:/Users/Dell/Desktop/face_recog/haarcascade_frontalface_default.xml')

def face_extractor(img):

    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_classifier.detectMultiScale(gray,1.3,5)
    
    if faces is():
       return None

    for(x,y,w,h) in faces:
        cropped_face=img[y:(y+h),x:(x+w)]

    return cropped_face
 

cap=cv2.VideoCapture(0,cv2.CAP_DSHOW)
count=0 

while True:
    ret,frame=cap.read()
    if face_extractor(frame) is not None:
        count+=1
        face=cv2.resize(face_extractor(frame),(200,200))
        face=cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)

        #writting all these photos 

        file_name_path='C:/Users/Dell/Desktop/face_recog/faces/user'+str(count)+'.jpg'
        cv2.imwrite(file_name_path,face)

        cv2.imshow('FACE CROPPER',face)

    else:
        print("face not find")
        pass
    if cv2.waitKey(1)==13 or count==100: #waits for enter
        break
cap.release()
cv2.destroyAllWindows()
print('collecting samples')