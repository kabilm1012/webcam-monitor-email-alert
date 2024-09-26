import cv2 as cv
import time
from emailing import send_email

video = cv.VideoCapture(0)
time.sleep(1)
first_frame = None
status_list = []

while True:
    status = 0
    check, frame = video.read()
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray_frame_gau = cv.GaussianBlur(gray_frame, (11, 11), 0)

    if first_frame is None:
        first_frame = gray_frame_gau
    
    delta_frame = cv.absdiff(gray_frame_gau, first_frame)
    thresh_frame = cv.threshold(delta_frame, 30, 255, cv.THRESH_BINARY)[1]
    dil_frame = cv.dilate(thresh_frame, None, iterations=2)

    cv.imshow("My Camera", dil_frame)

    contours, check = cv.findContours(dil_frame, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv.contourArea(contour) < 10000:
            continue
        x, y, w, h = cv.boundingRect(contour)
        rectangle = cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
        if rectangle.any():
            status = 1
        
    status_list.append(status)
    status_list = status_list[-2:]
    if status_list[0] == 1 and status_list[1] == 0:
        send_email()
    print(status_list)

    cv.imshow("Video", frame)
    key = cv.waitKey(1)

    if key == ord("q"):
        break

video.release()