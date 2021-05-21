"""
This program detect faces on a picture
by using openCV
"""

import cv2
from random import randrange

trained_face_data = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# get img file
img = cv2.imread('./maroon5.png')

# make the picture gray
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# detect faces
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

# draw rectangle around the faces
for (x,y,w,h) in face_coordinates:
    cv2.rectangle(img, (x, y), (x+w, y+h), (randrange(255), randrange(255), randrange(255)), 3)

# show the img with rectangles cover the faces
cv2.imshow('Clever Prgrammer Face Detector', img)

cv2.waitKey()

