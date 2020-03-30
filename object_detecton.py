import cv2
import numpy as np


def nothing(x):
    pass


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')



cap = cv2.VideoCapture(0)

cv2.namedWindow('trackbar')
cv2.createTrackbar('L-H', 'trackbar', 0, 180, nothing)
cv2.createTrackbar('L-S', 'trackbar', 123, 255, nothing)
cv2.createTrackbar('L-V', 'trackbar', 123, 255, nothing)
cv2.createTrackbar('U-H', 'trackbar', 180, 180, nothing)
cv2.createTrackbar('U-S', 'trackbar', 255, 255, nothing)
cv2.createTrackbar('U-V', 'trackbar', 255, 255, nothing)

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.1,4)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # defining the lower and upper parameters for red objectand changing them

    l_h = cv2.getTrackbarPos('L-H', 'trackbar')
    l_s = cv2.getTrackbarPos('L-S', 'trackbar')
    l_v = cv2.getTrackbarPos('L-V', 'trackbar')
    u_h = cv2.getTrackbarPos('U-H', 'trackbar')
    u_s = cv2.getTrackbarPos('U-S', 'trackbar')
    u_v = cv2.getTrackbarPos('U-V', 'trackbar')
    font = cv2.FONT_HERSHEY_COMPLEX

    lower_yellow = np.array([l_h, l_s, l_v])
    upper_yellow = np.array([u_h, u_s, u_v])

    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel)

    # contour detection

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.putText(frame, "face", (x, y), font, 1, (255,0,0), 2)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
        x = approx.ravel()[0]
        y = approx.ravel()[1]

        print(x)

        if area > 400:
            cv2.drawContours(frame, [approx], 0, (116.09, 51.57, 43.73), 3)
            if len(approx) == 4:
                cv2.putText(frame, "book", (x, y), font, 1, (116.09, 51.57, 43.73), 3)
            elif (10 < len(approx) < 20) & (x < 200):
                cv2.putText(frame, "Cap->LEFT", (x, y), font, 1, (116.09, 51.57, 43.73), 2)
            elif (10 < len(approx) < 20) & (x > 320):
                cv2.putText(frame, "Cap->RIGHT", (x, y), font, 1, (116.09, 51.57, 43.73), 2)
            elif (10 < len(approx) < 20) & (290 < x < 320):
                cv2.putText(frame, "Cap->CENTER", (x, y), font, 1, (116.09, 51.57, 43.73), 2)

    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
