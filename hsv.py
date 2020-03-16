import cv2
import numpy as np 


# cv2.imshow('img',img)
# cv2.waitKey(0)

# cv2.imshow('hsv',hsv)
def hsv_yellow():
    #橙黄色
    img = cv2.imread('E:\\paper-hsv\\1.png')
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    lower_hsv = np.array([11, 50, 80],dtype=np.uint8)
    #橙色HSV范围下限 原：11,43,46

    upper_hsv = np.array([19, 255, 255],dtype=np.uint8) 
    #橙色HSV范围上限 原：25,255,255
   
    mask = cv2.inRange(hsv,lower_hsv,upper_hsv)

    cv2.imshow('yellow',mask)
    cv2.waitKey(0)

def hsv_red():
    #橙黄色
    img = cv2.imread('E:\\paper-hsv\\1.png')
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    lower_hsv = np.array([0, 43, 46],dtype=np.uint8)
    #橙色HSV范围下限 原：0,43,46

    upper_hsv = np.array([10,255,255],dtype=np.uint8) 
    #橙色HSV范围上限 原：10,255,255
   
    mask = cv2.inRange(hsv,lower_hsv,upper_hsv)

    cv2.imshow('red',mask)
    cv2.waitKey(0)

def hsv_blue():
    #橙黄色
    img = cv2.imread('E:\\paper-hsv\\1.png')
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    lower_hsv = np.array([100,43,46],dtype=np.uint8)
    #橙色HSV范围下限 原：100,43,46

    upper_hsv = np.array([124, 255, 255],dtype=np.uint8) 
    #橙色HSV范围上限 原：124,255,255
   
    mask = cv2.inRange(hsv,lower_hsv,upper_hsv)

    cv2.imshow('blue',mask)
    cv2.waitKey(0)


if __name__ == '__main__':
    hsv_blue()
    