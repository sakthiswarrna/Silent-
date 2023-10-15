import cv2
import numpy as np
import math
import time
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import pyttsx3

cap = cv2.VideoCapture(0)

text = pyttsx3.init()
text.setProperty('rate',150)
offset = 20
imgSize = 300

detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model3.h5","Model/labels3.txt")



folder ="Data/V"
counter =0

labels = ["I AM","FINE","SEE","SILENT","A","V"]

if __name__ == '__main__':
    while True:
        success, img = cap.read()
        imgOutput = img.copy()

        hands, img = detector.findHands(img)

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
            imgCropShape = imgCrop.shape


            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((300 - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
                prediction,index = classifier.getPrediction(imgWhite,draw=False)



                print(prediction,index)

            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (hCal, imgSize))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((300 - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize
                prediction, index = classifier.getPrediction(imgWhite,draw=False)


                print(prediction, index)


            cv2.rectangle(imgOutput,(x-offset,y-offset-50),(x-offset+190,y-offset),(255,0,255),cv2.FILLED)
            cv2.putText(imgOutput,labels[index],(x,y-30),cv2.FONT_HERSHEY_COMPLEX,1.7,(255,255,255),2)
            cv2.rectangle(imgOutput,(x-offset,y-offset),(x+w+offset,y+h+offset),(255,0,255),4)

            ans = labels[index]
            text.say(ans)
            text.runAndWait()

            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite)

        cv2.imshow("Images", imgOutput)
        key = cv2.waitKey(1)

        if key == ord("s"):
            counter += 1
            cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
            print(counter)


