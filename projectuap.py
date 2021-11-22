import cv2
import os
import numpy as np 
import math

hair_face = cv2.CascadeClassifier("Had.xml")
trainpath = "dataset/train"
member_list = os.listdir(trainpath)


list_wajah = []
member_tag = []

for i,member in enumerate(member_list):
    member_path = trainpath + "/" + member
    for gambar in os.listdir(member_path):
        gambar_member_path = member_path + "/" + gambar
        gambar_asli = cv2.imread(gambar_member_path)
        gambar_gray = cv2.cvtColor(gambar_asli,cv2.COLOR_BGR2GRAY)

        face_train = hair_face.detectMultiScale(gambar_gray, scaleFactor = 1.2,minNeighbors = 5)
        if(len(face_train) < 1):
            continue

        for titik in face_train:
            x,y,w,h = titik
            wajah = gambar_gray[y:y+h, x:x+w]
            list_wajah.append(wajah)
            member_tag.append(i)

# for image in list_wajah :
#     cv2.imshow("esult",image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows
face_recognizer = cv2.face.LBPHFaceRecognizer_create() 

face_recognizer.train(list_wajah,np.array(member_tag))

testpath = "Test10.jpg"

image = cv2.imread(testpath)

gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)


face = hair_face.detectMultiScale(gray, scaleFactor=1.2, minNeighbors = 10)


# for titik in face_train:
#     x,y,w,h = titik
#     wajah = gambar_gray[y:y+h, x:x+w]
#     list_wajah.append(wajah)
#     member_tag.append(i)

wajah_test_tag = []
wajah_test_list = []
huruf = cv2.FONT_HERSHEY_TRIPLEX
ukuran_huruf = 1.0
warna_huruf =(0,0,255)


for face_react in face :
    x,y,w,h = face_react
    face_img = gray[y:y+h, x:x+w]
    res, confindence = face_recognizer.predict(face_img)
    confindence= math.floor(confindence*100) /100
    cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),5)
    text = member_list[res] + " " + str(confindence) + '%'
    cv2.putText(image,text,(x,y-10),huruf,ukuran_huruf,warna_huruf)
    

    

    
# face = hair_face.detectMultiScale(gray, scaleFactor=1.2, minNeighbors = 10) -> test wajah 4


# print("Face Found ", len(face))




# wajah_test_list = []

# for x,w,y,h in face :
#     cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),5)
#     wajah_test = image[y:y+h, x:x+w]       
#     wajah_test_list.append(wajah_test)             

# for image in wajah_test_list:
#     cv2.imshow("Result",image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()                                        
                                        #bgr

cv2.imshow("image",image)
cv2.waitKey(0)
cv2.destroyAllWindows()
