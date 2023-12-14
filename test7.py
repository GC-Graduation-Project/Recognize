import os
import cv2
import numpy as np
from PIL import Image
from pytesseract import *

import modules

resource_path = os.getcwd() + "/resources/"
img = cv2.imread(resource_path + "rotate1.jpg")
draw = img.copy()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray,(3,3), 0)
edged = cv2.Canny(gray, 75, 200)
cv2.imshow('edge',edged)
cv2.waitKey(0)
( cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(len(cnts))
cv2.drawContours(draw, cnts, -1,(0,255,0))

cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
for c in cnts:
    peri = cv2.arcLength(c,True)
    verticles = cv2.approxPolyDP(c,0.02*peri,True)
    if len(verticles)==4:
        break
pts = verticles.reshape(4,2)
for x, y in pts:
    cv2.circle(draw, (x,y), 10,(0,255,0),-1)

sm = pts.sum(axis=1)
diff = np.diff(pts, axis=1)
topLeft = pts[np.argmin(sm)]
bottomRight = pts[np.argmax(sm)]
topRight = pts[np.argmin(diff)]
bottomLeft = pts[np.argmax(diff)]

pts1 = np.float32([topLeft, topRight, bottomRight, bottomLeft])
w1 = abs(bottomRight[0]-bottomLeft[0])
w2 = abs(topRight[0]-topLeft[0])
h1 = abs(topRight[1]-bottomRight[1])
h2 = abs(topLeft[1]-bottomLeft[1])
width = max([w1, w2])
height = max([h1, h2])

pts2 = np.float32([[0, 0], [width -1, 0],[width -1, height-1],[0, height-1]])

mtrx = cv2.getPerspectiveTransform(pts1,pts2)
result = cv2.warpPerspective(img,mtrx,(width, height))


cv2.imshow('win_name', result)
cv2.waitKey(0)