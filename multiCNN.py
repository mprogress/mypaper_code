
import os
import numpy as np
import PIL
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import cv2
 
train_dir = 'E:\\paper-hsv\\training'
GT_dir = 'E:\\paper-hsv\\GT_save'

'''
#读取训练集txt的内容：名称；坐标；标志类型
def convert_train_data(file_dir):
 



#计算面积的交并集IoU
def calculate_IoU(predicted_bound, ground_truth_bound):
    """
    computing the IoU of two boxes.
    Args:
        box: (xmin, ymin, xmax, ymax),通过左下和右上两个顶点坐标来确定矩形位置
    Return:
        IoU: IoU of box1 and box2.
    """
    pxmin, pymin, pxmax, pymax = predicted_bound
    print("预测框P的坐标是：({}, {}, {}, {})".format(pxmin, pymin, pxmax, pymax))
    gxmin, gymin, gxmax, gymax = ground_truth_bound
    print("原标记框G的坐标是：({}, {}, {}, {})".format(gxmin, gymin, gxmax, gymax))

    parea = (pxmax - pxmin) * (pymax - pymin)  # 计算P的面积
    garea = (gxmax - gxmin) * (gymax - gymin)  # 计算G的面积
    print("预测框P的面积是：{}；原标记框G的面积是：{}".format(parea, garea))

    # 求相交矩形的左下和右上顶点坐标(xmin, ymin, xmax, ymax)
    xmin = max(pxmin, gxmin)  # 得到左下顶点的横坐标
    ymin = max(pymin, gymin)  # 得到左下顶点的纵坐标
    xmax = min(pxmax, gxmax)  # 得到右上顶点的横坐标
    ymax = min(pymax, gymax)  # 得到右上顶点的纵坐标

    # 计算相交矩形的面积
    w = xmax - xmin
    h = ymax - ymin
    if w <=0 or h <= 0:
        return 0

    area = w * h  # G∩P的面积
    # area = max(0, xmax - xmin) * max(0, ymax - ymin)  # 可以用一行代码算出来相交矩形的面积
    print("G∩P的面积是：{}".format(area))

    # 并集的面积 = 两个矩形面积 - 交集面积
    IoU = area / (parea + garea - area)

    return IoU
'''
if __name__ == '__main__':
    # IoU = calculate_IoU( (1, -1, 3, 1), (0, 0, 2, 2))
    # print("IoU是：{}".format(IoU))
    ###############################################  
    #          
    aa = []
    txt_data = pd.read_csv('E:\\paper-hsv\\GroundTruth.txt')
    # csv_data = pd.read_csv(csv_dir)
    txt_data_array = np.array(txt_data)
    # print(csv_data_array)
    for i in range(txt_data_array.shape[0]):
            txt_data_list = np.array(txt_data)[i,:].tolist()[0].split(";")
            print(txt_data_list)
            # ['00000_00000.ppm', '29', '30', '5', '6', '24', '25', '0']
            # ['00000_00001.ppm', '30', '30', '5', '5', '25', '25', '0']

            sample_dir = os.path.join(train_dir, txt_data_list[0])
            # 获取该data_dir目录下每张图片的绝对地址
            # print(sample_dir)，部分展示如下：
            # E:\DataSet\GTRSB\GTSRB_Final_Training_Images\GTSRB\Final_Training\Images\00000\00000_00000.ppm
 
 
            img = PIL.Image.open(sample_dir)
            # img = cv2.imread('%s'%(sample_dir))
            res = img.copy()
            box = (int(txt_data_list[1]),int(txt_data_list[2]),int(txt_data_list[3]),int(txt_data_list[4]))
            roi_img = res.crop(box)
            # 获取兴趣ROI区域

            cv2.rectangle(res,(txt_data_list[1],txt_data_list[2]),(txt_data_list[3],txt_data_list[4]),(0,255,0),2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = 'GT'
            cv2.putText(res, text, (txt_data_list[1]-30,txt_data_list[2]), cv2.FONT_HERSHEY_COMPLEX, 20, (0,255,0), 2)
            #cv2.putText(img, str,origin,font,size,color,thickness)

            # cv2.imwrite('%s\\%s'%(GT_dir,txt_data_list[0]), res)
     
            #显示画了标志的原图       
            cv2.imshow('res',res)
            cv2.waitKey(0)
