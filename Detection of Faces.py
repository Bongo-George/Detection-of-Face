import cv2
from random import randrange
trained_face_data = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
#img = cv2.imread("lena.jpg")
#img = cv2.imread("RDJ.jpeg")
#img = cv2.imread("boy.jpg")
webcam = cv2.VideoCapture(0)

while True:
    successful_frame_read, frame = webcam.read()

    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (randrange(256), randrange(255), randrange(256)), 3)
    cv2.imshow("Face Detection", frame)
    key = cv2.waitKey(1)
    if key == 81 or key == 113:
        break
webcam.release()

""""
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

for (x, y, w, h) in face_coordinates:
#(x, y, w, h) = face_coordinates[0]
#x,y x+w,y+h
    cv2.rectangle(img, (x, y), (x+w, y+h), (randrange(256), randrange(255), randrange(256)), 3)

cv2.imshow("Bongo George Face Detection", img)
cv2.waitKey()"""