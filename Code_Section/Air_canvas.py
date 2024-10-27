import cv2
import numpy as np
import time
import os
import Track_Hands_Module as htm
from tensorflow.tools.docs import doc_controls

eraserThickness = 50
brushThickness = 15
folderPath = "header"
mylist = os.listdir(folderPath)
# print(mylist)
overlayList = []

for imPath in mylist:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)

# print(len(overlayList))
header = overlayList[0]
drawColour = (203, 189, 253)  # bgr format

cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Set the width
cap.set(4, 720)  # Set thr height
detector = htm.HandDetector(detectionCon=0.85)

xp, yp = 0, 0

imageCanvas = np.zeros((480, 640, 3), np.uint8)  # height,width,no.ofchannels

while True:
    # Import the image
    success, img = cap.read()

    img = cv2.flip(img, 1)

    # Find the hand Landmarks

    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:

        # print(lmList)

        # Coordinates of the tip of our index finger (lmlist mein 1st and 2nd index of the 8h index are the x and y coordinates)
        x1, y1 = lmList[8][1:]
        # #Coordinates of the tip of our middle finger (lmlist mein 1st and 2nd index of the 8h index are the x and y coordinates)
        x2, y2 = lmList[12][1:]

        # Check which fingers are up

        fingers = detector.fingersUp()
        # print(fingers)

        # If selection mode - two fingers are up

        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            # print("selection mode")
            if y1 < 125:
                if 135 < x1 < 215:
                    header = overlayList[0]
                    drawColour = (203, 189, 253)
                elif 216 < x1 < 296:
                    header = overlayList[1]
                    drawColour = (255, 0, 0)
                elif 310 < x1 < 395:
                    header = overlayList[2]
                    drawColour = (0, 255, 0)
                elif 410 < x1 < 535:
                    header = overlayList[3]
                    drawColour = (0, 0, 0)
                elif 537 < x1 < 640:
                    header = overlayList[4]
                    drawColour = (255, 255, 255)
                    break
            cv2.rectangle(img, (x1, y1-25), (x2, y2+25),
                          drawColour, cv2.FILLED)

        # If drawing mode - index finger is up
        elif fingers[1] and fingers[2] == False:
            cv2.circle(img, (x1, y1), 15, drawColour, cv2.FILLED)
            # print("Drawing mode")

            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            if drawColour == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), drawColour, eraserThickness)
                cv2.line(imageCanvas, (xp, yp), (x1, y1),
                         drawColour, eraserThickness)
            else:
                cv2.line(img, (xp, yp), (x1, y1), drawColour, brushThickness)
                cv2.line(imageCanvas, (xp, yp), (x1, y1),
                         drawColour, brushThickness)

            xp, yp = x1, y1  # the next moment the current point becomes the previous point

    # Converting the imagecanvas to an gray image
    imageGray = cv2.cvtColor(imageCanvas, cv2.COLOR_BGR2GRAY)

    # Convert the gray image to a binary and inversed image
    _, imageInverse = cv2.threshold(imageGray, 50, 255, cv2.THRESH_BINARY_INV)
    # WE can't add a gray image to a coloured image
    imageInverse = cv2.cvtColor(imageInverse, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imageInverse)
    img = cv2.bitwise_or(img, imageCanvas)

    # setting the header image
    img[0:125, 0:640] = header

    # Adding two images

    # img = cv2.addWeighted(img,0.5,imageCanvas,0.5,0)    #This is just a blend(there will be transparency and colours won't be that bright)
    cv2.imshow("image", img)
    cv2.imshow("ImageCanvas", imageCanvas)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()

print("**************************************THANKYOU FOR USING OUR AIR CANVAS**************************************")
