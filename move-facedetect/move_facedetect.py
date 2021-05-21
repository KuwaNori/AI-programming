import cv2
from random import randrange

trained_face_data = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

webcam = cv2.VideoCapture(0)

while True:
    successful_frame_read, frame = webcam.read()

    grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_frame)

    # draw rectangle around the faces
    for (x,y,w,h) in face_coordinates:

        cv2.rectangle(frame, (x, y), (x+w, y+h), (randrange(255), randrange(255), randrange(255)), 3)

# show the img with rectangles cover the faces
    cv2.imshow('Clever Prgrammer Face Detector', frame)

    cv2.waitKey(1)
