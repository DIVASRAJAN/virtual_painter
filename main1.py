import cv2
import HandTrackingModule as htm
import numpy as np
import time

detector = htm.handDetector()

draw_color = (0, 0, 255)

img_canvas = np.zeros((720, 1280, 3), np.uint8)
p_time =0

cap = cv2.VideoCapture(0)

while True:

    success, frame = cap.read()
    frame = cv2.resize(frame, (1280, 720))
    frame = cv2.flip(frame, 1)
    cv2.rectangle(frame, (10, 10), (250, 100), (255, 0, 0), cv2.FILLED)
    cv2.rectangle(frame, (260, 10), (500, 100), (0, 255, 0), cv2.FILLED)
    cv2.rectangle(frame, (510, 10), (750, 100), (0, 0, 255), cv2.FILLED)
    cv2.rectangle(frame, (760, 10), (1000, 100), (255, 255, 0), cv2.FILLED)
    cv2.rectangle(frame, (1010, 10), (1270, 100), (255, 255, 255), cv2.FILLED)
    cv2.putText(frame, 'ERASER', (1071, 63), fontScale=1, fontFace=cv2.FONT_HERSHEY_COMPLEX, color=(0, 0, 0),
                thickness=4)

    # find hand landmarks
    frame = detector.findHands(frame, draw=True)
    lmlist = detector.findPosition(frame)  # which is the coordination of the points
    # print(lmlist)

    if len(lmlist) != 0:
        x1, y1 = lmlist[8][1:]  # index finger coordinates
        x2, y2 = lmlist[12][1:]  # middle finger coordinates
        # print(x1,y1)

        # check which finger is up
        fingers = detector.fingersUp()
        # print(fingers)

        # selection mode - two fingers is up

        if fingers[1] and fingers[2]:
            # print('selection mode')
            xp,yp = 0,0

            if y1 < 100:
                if 10 < x1 < 250:
                    draw_color = (250, 0, 0)
                    # print('blue')

                elif 260 < x1 < 500:
                    draw_color = (0, 255, 0)
                    # print('green')

                elif 510 < x1 < 750:
                    draw_color = (0, 0, 255)
                    # print('red')

                elif 760 < x1 < 1000:
                    draw_color = (255, 255, 0)
                    # print('cyan')

                elif 1010 < x1 < 1270:
                    draw_color = (0, 0, 0)
                    # print('eraser')

            cv2.rectangle(frame, (x1, y1), (x2, y2), draw_color, thickness=cv2.FILLED)

        # drawing mode - index finger is up

        if (fingers[1] and not fingers[2]):
            # print('drawing mode')

            cv2.putText(frame, 'Drawing Mode', (1040, 670), fontFace=cv2.FONT_HERSHEY_COMPLEX, color=(255, 0, 255),
                        fontScale=1, thickness=3)
            cv2.circle(frame, (x1, y1), 10, draw_color, thickness=-1)

            if xp==0 and yp == 0:
                xp=x1
                yp=y1

            if draw_color == (0,0,0):
                cv2.line(frame,(xp,yp),(x1,y1),color=draw_color,thickness=20)
                cv2.line(img_canvas, (xp, yp), (x1, y1), color=draw_color, thickness=20)

            else:
                cv2.line(frame,(xp,yp),(x1,y1),color=draw_color,thickness=10)
                cv2.line(img_canvas, (xp, yp), (x1, y1), color=draw_color, thickness=10)

            xp,yp=x1,y1

    # merging two canvas
    img_gray = cv2.cvtColor(img_canvas,cv2.COLOR_BGR2GRAY)
    # thresh inverse
    thresh,img_inv = cv2.threshold(img_gray,20,255,cv2.THRESH_BINARY_INV) # below 20 white and above black
    img_inv =cv2.cvtColor(img_inv,cv2.COLOR_GRAY2BGR)

    # and operation

    frame = cv2.bitwise_and(frame,img_inv)
    frame = cv2.bitwise_or(frame,img_canvas)

    frame = cv2.addWeighted(frame,1,img_canvas,0.5,0)

    # calculating fps

    c_time = time.time()
    fps= 1/(c_time-p_time)
    p_time = c_time
    cv2.putText(frame,str(int(fps)),org=(50,150) ,fontScale=1, fontFace=cv2.FONT_HERSHEY_COMPLEX, color=(0, 255, 0),
                thickness=4)

    cv2.imshow('virtual painter', frame)
    # cv2.imshow('canvas', img_canvas)
    # cv2.imshow('painter', img_inv)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()

