import time
import numpy as np
import cv2 as cv
import HandTrackingModule as htm
import math

ptime = 0
ctime = 0
wcam, hcam = 640, 480
cap = cv.VideoCapture(0)
cap.set(3, wcam)
cap.set(4, hcam)
detector = htm.handDetector(detectionCon=0.7)


# ========================================================================= #
# pycaw library installed from GitHub
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
volume.GetMute()
volume.GetMasterVolumeLevel()
volumeRange=volume.GetVolumeRange()
# print(volumeRange)
minVol=volumeRange[0]
maxVol=volumeRange[1]
# ========================================================================= #


vol=0
volBar=400
volPer=0
while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmlist=detector.findPosition(img, draw=False)
    # print(lmlist)
    
    if len(lmlist)!=0:  #* -----> this is used here because if no hands are shown then the code will end since it will not be able to find any hand and hence the list will become empty
        # print(lmlist[4], lmlist[8])
        
        x1,y1=lmlist[4][1],lmlist[4][2]
        x2,y2=lmlist[8][1],lmlist[8][2]
        cx,cy=(x1+x2)//2, (y1+y2)//2
        
        cv.circle(img,(x1,y1),10,(0,255,255), cv.FILLED)
        cv.circle(img,(x2,y2),10,(0,255,255), cv.FILLED)
        cv.circle(img,(cx,cy),10,(0,255,255), cv.FILLED)
        cv.line(img, (x1,y1),(x2,y2),(0,255,255),4)
        
        ###############& Based on the length of the line we can change the volume accordingly###############
        length=math.hypot(x2-x1,y2-y1)
        if length<=50:
            cv.circle(img,(cx,cy),10,(255,255,0),cv.FILLED)
        # print(length)
        
        # Hand Range : 50  <--->  300
        # Volume Range : -96  <--->  0
        
        vol=np.interp(length,[50,150],[minVol,maxVol])
        volBar=np.interp(length,[50,150],[400,150])
        volPer=np.interp(length,[50,150],[0,100])
        volume.SetMasterVolumeLevel(vol, None)
        # print(vol)
    
    cv.rectangle(img,(50,150),(85,400),(0,255,0), 3)
    cv.rectangle(img,(50,int(volBar)),(85,400),(0,255,0), cv.FILLED)
    cv.putText(img, f'VOL: {int(volPer)} %', (40, 450), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 3)


    #########################################^ FPS Shower ###################################################
    ctime=time.time()
    fps=1/(ctime-ptime)
    ptime=ctime

    cv.putText(img, f'FPS: {int(fps)}', (20, 60), cv.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)

    cv.imshow("Img", img)
    cv.waitKey(1)