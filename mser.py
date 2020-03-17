import cv2
import matplotlib.pyplot as plt

img = cv2.imread('E:\\paper-hsv\\training\\3.png')
mask = cv2.imread('E:\\paper-hsv\\mask\\mask.png')
res = img.copy()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

mser = cv2.MSER_create(_min_area=100)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
regions, boxes = mser.detectRegions(gray)

for box in boxes:
    x, y, w, h = box
    cv2.rectangle(res, (x,y),(x+w, y+h), (0, 255, 0), 1)

cv2.imshow('res',res)
cv2.waitKey(0)
