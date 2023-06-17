import cv2
import time
import numpy as np
import HandTrackingModule as htm
import math
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)

volRange = volume.GetVolumeRange()

minVol = volRange[0]
maxVol = volRange[1]

wCam, hCam = 640, 480

cap = cv2.VideoCapture(1)
cap.set(3, wCam)
cap.set(4, hCam)

prevTime = 0

detector = htm.handDetector(detectionCon=0.7)

while True:
        success, img = cap.read()
        img = detector.findHands(img)

        lmList = detector.findPosition(img, draw=False)

        if len(lmList) > 0:
            x1, y1 = lmList[4][1], lmList[4][2]
            x2, y2 = lmList[8][1], lmList[8][2]

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            cv2.circle(img, (x1, y1), 15, (0,0,255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (0,0,255), cv2.FILLED)

            cv2.line(img, (x1, y1), (x2, y2), (0,0,255), 3)

            length = math.hypot(x2-x1, y2-y1)

            # Hand Range: 50 to 300
            # Volume Range: -65 to 0

            vol = np.interp(length, [50,300], [minVol, maxVol])
            
            volume.SetMasterVolumeLevel(vol, None)

            if length < 50:
                cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)                

        currTime = time.time()
        fps = 1 / (currTime - prevTime)
        prevTime = currTime

        cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 2) 

        cv2.imshow("Image", img) 
        cv2.waitKey(1) 

    