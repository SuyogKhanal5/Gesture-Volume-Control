import cv2
import mediapipe as mp
import time

class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands 
        self.hands = self.mpHands.Hands(static_image_mode=self.mode, max_num_hands=self.maxHands,
                                         min_detection_confidence=self.detectionCon, min_tracking_confidence=self.trackCon) 
        self.mpDraw = mp.solutions.drawing_utils 
        
    def findHands(self, img, draw=True):
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        self.results = self.hands.process(self.imgRGB)     

        if (self.results.multi_hand_landmarks): 
            for handLms in self.results.multi_hand_landmarks: 
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS) 

        return img

    def placeCircle(self, img, landmarkNum):
        if self.results.multi_hand_landmarks: 
            for handLms in self.results.multi_hand_landmarks: 
                for id, lm in enumerate(handLms.landmark): 
                    height, width, channels = img.shape

                    centerX, centerY = int(lm.x * width), int(lm.y * height)

                    if id == landmarkNum: 
                        cv2.circle(img, (centerX, centerY), 25, (0, 0, 255), cv2.FILLED)

        return img
    
    def findPosition(self, img, handNo=0, draw=True):
        lmList = []

        if self.results.multi_hand_landmarks: 
            chosenHand = self.results.multi_hand_landmarks[handNo]

            for id, lm in enumerate(chosenHand.landmark): 
                height, width, channels = img.shape

                centerX, centerY = int(lm.x * width), int(lm.y * height)

                lmList.append([id, centerX, centerY])

                if draw:
                    cv2.circle(img, (centerX, centerY), 15, (255, 0, 0), cv2.FILLED)

        return lmList   


def main():
    prevTime = 0
    currTime = 0

    cap = cv2.VideoCapture(1) 

    detector = handDetector()

    while True:
        success, img = cap.read()
        img = detector.findHands(img)  
        lmList = detector.findPosition(img)
        img = detector.placeCircle(img, 8)

        if (len(lmList) != 0):
            print(lmList[8])

        currTime = time.time()
        fps = 1 / (currTime - prevTime)
        prevTime = currTime

        cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 2) 

        cv2.imshow("Image", img) 
        cv2.waitKey(1) 


if __name__ == "__main__":
    main()