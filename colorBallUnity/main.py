import cvzone
import cv2
from cvzone.ColorModule import ColorFinder
import socket

cap = cv2.VideoCapture(0)
cap.set(3, 1080)
cap.set(4, 720)

success, img = cap.read()
h, w, _ = img.shape

myColorFinder = ColorFinder(False)
hsvVals = {'hmin': 55, 'smin': 110, 'vmin': 167, 'hmax': 179, 'smax': 255, 'vmax': 242}
# hsvVals = {'hmin': 10, 'smin': 55, 'vmin': 215, 'hmax': 42, 'smax': 255, 'vmax': 255}

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
serverAddressPort = ("127.0.0.1", 5053)
while True:
    success, img = cap.read()
    imgColor, mask = myColorFinder.update(img, hsvVals)
    imgContour, contours = cvzone.findContours(img, mask, minArea=1000)

    if contours:
        data = contours[0]['center'][0], \
               h - contours[0]['center'][1], \
               int(contours[0]['area'])
        print(data)
        sock.sendto(str.encode(str(data)), serverAddressPort)
    # imgStack = cvzone.stackImages([img, imgColor,mask, imgContour], 2, 0.5)# mask, imgContour
    # cv2.imshow("Image", imgStack)
    imgContour = cv2.resize(imgContour, (0, 0), None, 0.6, 0.6)
    cv2.imshow("Image", imgContour)
    cv2.waitKey(1)
