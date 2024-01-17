import cv2 as cv
import mediapipe as mp
import numpy as np
import time

mpHands=mp.solutions.hands
hands=mpHands.Hands()
mpdraw=mp.solutions.drawing_utils

wcam,hcam=640,480

cap=cv.VideoCapture(0)
cap.set(3, wcam)
cap.set(4, hcam)
ptime=0

while True:
    success,img=cap.read()
    imgRGB=cv.cvtColor(img,cv.COLOR_BGR2RGB)
    results=hands.process(imgRGB)
    palms=results.multi_hand_landmarks
    # print(results.multi_hand_landmarks) ----> this shows the position of the hands in the camera
    
    if palms:
        for handLms in palms:
            for id,lm in enumerate(handLms.landmark):
                # print(id,lm) ----> This would print the coordinates of the landmarks of all 21 points at the hand in one instance
                h,w,c=img.shape
                cx,cy=int(lm.x*w),int(lm.y*h)
                # print(id,cx,cy) # -----> This prints the points in pixels along with the id of the landmarks point
                if id==0:
                    cv.circle(img,(cx,cy),15,(255,0,255),cv.FILLED)
                elif id==4:
                    cv.circle(img,(cx,cy),15,(255,255,0),cv.FILLED)
            mpdraw.draw_landmarks(img,handLms,mpHands.HAND_CONNECTIONS)
    
    ctime=time.time()
    fps=1/(ctime-ptime)
    ptime=ctime
    
    cv.putText(img,f'FPS: {int(fps)}',(20,60),cv.FONT_HERSHEY_COMPLEX,1,(255,0,0),3)
    
    cv.imshow("Img",img)
    cv.waitKey(1)