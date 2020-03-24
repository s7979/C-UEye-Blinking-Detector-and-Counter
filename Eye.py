import cv2
import numpy as np

import dlib
from math import hypot

cap  = cv2.VideoCapture(0) #0 one or more webcam k liy

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("Shape_predictor_68_face_landmarks.dat")

TOTAL=0

def midpoint(p1,p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

font = cv2.FONT_HERSHEY_PLAIN

def Eye_aspect_ratio(Eye_points,Facial_landmarks):
    left_point = (Facial_landmarks.part(Eye_points[0]).x, Facial_landmarks.part(Eye_points[1]).y)
    right_point = (Facial_landmarks.part(Eye_points[3]).x, Facial_landmarks.part(Eye_points[3]).y)
    center_top = midpoint(Facial_landmarks.part(Eye_points[1]), Facial_landmarks.part(Eye_points[2]))
    center_bottom = midpoint(Facial_landmarks.part(Eye_points[5]), Facial_landmarks.part(Eye_points[4])) 

    hor_line = cv2.line(frame, left_point, right_point,(0,255,0),2)
    ver_line = cv2.line(frame, center_top, center_bottom, (0,255,0), 2)
        
    hor_line_lenght = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_lenght = hypot((center_top[0]- center_bottom[0]), (center_top[1] - center_bottom[1])) 

    ratio=hor_line_lenght / ver_line_lenght
    
    return ratio
while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)# converting bgr image to gray because gray is easy for computational purpose

    faces = detector(gray)
    for  face in faces:
        #x1,y1 = face.left(), face.top()
        #x2,y2 = face.right(), face.bottom()
        #cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2) #used tto draw rectangle with coordi's and RGB format 0,255,0 and thickness parameter 2
        
        landmarks = predictor(gray,face)
         
        left_eye_ratio = Eye_aspect_ratio([36, 37, 38, 39, 40, 41],landmarks)
        right_eye_ratio = Eye_aspect_ratio([42, 43, 44, 45, 46, 47],landmarks)
        blinking_ratio = (left_eye_ratio+right_eye_ratio)/2

        
        if blinking_ratio > 5.7:
            TOTAL += 1
            cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
			    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        
        
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break
cap.release()
cv2.destroyAllwindows()