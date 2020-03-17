import cv2
import numpy as np 
import os


# cv2.imshow('img',img)
# cv2.waitKey(0)

# cv2.imshow('hsv',hsv)

training_dir = 'E:\\paper-hsv\\training'
mask_dir = 'E:\\paper-hsv\\mask' 
file_names = [os.path.join(training_dir, f) for f in os.listdir(training_dir)  if f.endswith(".png")]

## file_name里面每个元素都是以.ppm为后缀的文件的绝对地址
# for file_name in file_names:
#     # print(file_name)
#     img = cv2.imread('%s'%(file_name))


def hsv_yellow():
    ###############橙黄色
    for file_name in file_names:
        # print(file_name)
        img = cv2.imread('%s'%(file_name))
        # img = cv2.imread('E:\\paper-hsv\\training')
        hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        lower_hsv_yellow = np.array([11, 43, 46],dtype=np.uint8)
        #橙色HSV范围下限 原：11,43,46

        upper_hsv_yellow = np.array([19, 255, 255],dtype=np.uint8) 
        #橙色HSV范围上限 原：25,255,255
    
        mask_yellow = cv2.inRange(hsv,lower_hsv_yellow,upper_hsv_yellow)
        # cv2.imshow('yellow',mask)
        # cv2.waitKey(0)
        ########优化处理mask区域###################
        #模糊
        blurred = cv2.blur(mask_yellow,(9,9))
        # cv2.imshow('blurred',blurred)
        
        #二值化
        ret,binary = cv2.threshold(blurred,127,255,cv2.THRESH_BINARY)
        # cv2.imshow('blurred binary',binary)
        # cv2.waitKey(0)

        #闭运算——使区域闭合无空隙
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        # cv2.imshow('closed',closed)
        # cv2.waitKey(0)

        (filepath, tempfilename) = os.path.split(file_name)
        mask_name = tempfilename.split('.')[0]
        # print(mask_name)
       
        cv2.imwrite('%s\\%s_mask.png'%(mask_dir,mask_name), closed)

        ##############如果场景比较复杂，仍然存在颜色干扰，可以采用膨胀和腐蚀操作进行去除干扰。##############
        #腐蚀和膨胀
        '''
        腐蚀操作将会腐蚀图像中白色像素，以此来消除小斑点，
        而膨胀操作将使剩余的白色像素扩张并重新增长回去。
        '''
        erode = cv2.erode(closed,None,iterations=4)
        # cv2.imshow('erode',erode)
        mask = cv2.dilate(erode,None,iterations=4)
        # cv2.imshow('dilate',dilate)
        # cv2.waitKey(0)



def hsv_red():
    #红色
    for file_name in file_names:
        # print(file_name)
        img = cv2.imread('%s'%(file_name))
        # img = cv2.imread('E:\\paper-hsv\\training')
        hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

        lower_hsv = np.array([0, 43, 46],dtype=np.uint8)
        #红色HSV范围下限 原：0,43,46

        upper_hsv = np.array([10,255,255],dtype=np.uint8) 
        #红色HSV范围上限 原：10,255,255
    
        mask = cv2.inRange(hsv,lower_hsv,upper_hsv)

        cv2.imshow('red',mask)
        cv2.waitKey(0)

        ########优化处理mask区域###################
        #模糊
        blurred = cv2.blur(mask,(9,9))
        # cv2.imshow('blurred',blurred)
        
        #二值化
        ret,binary = cv2.threshold(blurred,127,255,cv2.THRESH_BINARY)
        # cv2.imshow('blurred binary',binary)
        # cv2.waitKey(0)

        #闭运算——使区域闭合无空隙
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        # cv2.imshow('closed',closed)
        # cv2.waitKey(0)
        
        cv2.imwrite('E:\\paper-hsv\\mask\\mask.png', closed)

        ##############如果场景比较复杂，仍然存在颜色干扰，可以采用膨胀和腐蚀操作进行去除干扰。##############
        #腐蚀和膨胀
        '''
        腐蚀操作将会腐蚀图像中白色像素，以此来消除小斑点，
        而膨胀操作将使剩余的白色像素扩张并重新增长回去。
        '''
        # erode = cv2.erode(closed,None,iterations=4)
        # # cv2.imshow('erode',erode)
        # mask = cv2.dilate(erode,None,iterations=4)
        # # cv2.imshow('dilate',dilate)
        # # cv2.waitKey(0)
    

def hsv_blue():
    #蓝色
    for file_name in file_names:
        # print(file_name)
        img = cv2.imread('%s'%(file_name))
        # img = cv2.imread('E:\\paper-hsv\\training')
        hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

        lower_hsv = np.array([100,43,46],dtype=np.uint8)
        #蓝色HSV范围下限 原：100,43,46

        upper_hsv = np.array([124, 255, 255],dtype=np.uint8) 
        #蓝色HSV范围上限 原：124,255,255
    
        mask = cv2.inRange(hsv,lower_hsv,upper_hsv)

        cv2.imshow('red',mask)
        cv2.waitKey(0)

        ########优化处理mask区域###################
        #模糊
        blurred = cv2.blur(mask,(9,9))
        # cv2.imshow('blurred',blurred)
        
        #二值化
        ret,binary = cv2.threshold(blurred,127,255,cv2.THRESH_BINARY)
        # cv2.imshow('blurred binary',binary)
        # cv2.waitKey(0)

        #闭运算——使区域闭合无空隙
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        # cv2.imshow('closed',closed)
        # cv2.waitKey(0)
        
        cv2.imwrite('E:\\paper-hsv\\mask\\mask.png', closed)

        ##############如果场景比较复杂，仍然存在颜色干扰，可以采用膨胀和腐蚀操作进行去除干扰。##############
        #腐蚀和膨胀
        '''
        腐蚀操作将会腐蚀图像中白色像素，以此来消除小斑点，
        而膨胀操作将使剩余的白色像素扩张并重新增长回去。
        '''
        # erode = cv2.erode(closed,None,iterations=4)
        # # cv2.imshow('erode',erode)
        # mask = cv2.dilate(erode,None,iterations=4)
        # # cv2.imshow('dilate',dilate)
        # # cv2.waitKey(0)



####裁剪ROI##########
def cut_ROI(mask):
    # 查找轮廓
    contours, hierarchy = cv2.findContours(mask , cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)
    print('counts:',len(contours))
    i = 0
    for file_name in file_names:
        # print(file_name)
        img = cv2.imread('%s'%(file_name))
        res = img.copy()
        for con in contours:
            #轮廓转换为矩形
            rect = cv2.minAreaRect(con)
            #矩形转换为box
            box = np.int0(cv2.boxPoints(rect))
            #画出目标区域
            cv2.drawContours(res,[box],-1,(0,255,0),1)
            print([box])
            #计算矩形的行列
            h1=max([box][0][0][1],[box][0][1][1],[box][0][2][1],[box][0][3][1])
            h2=min([box][0][0][1],[box][0][1][1],[box][0][2][1],[box][0][3][1])
            l1=max([box][0][0][0],[box][0][1][0],[box][0][2][0],[box][0][3][0])
            l2=min([box][0][0][0],[box][0][1][0],[box][0][2][0],[box][0][3][0])
            print('h1',h1)
            print('h2',h2)
            print('l1',l1)
            print('l2',l2)
            #加上防错处理，确保裁剪区域无异常
            if h1-h2>0 and l1-l2>0:
                #裁剪矩形区域
                temp=mask[h2:h1,l2:l1]
                i=i+1
                #显示裁剪后的标志
                # cv2.imshow('sign'+str(i),temp)
        #显示画了标志的原图       
        cv2.imshow('res',res)

        cv2.waitKey(0)


if __name__ == '__main__':
    hsv_yellow()
    
    for masks in os.listdir(mask_dir):
        # print(masks)  
        mask = cv2.imread(mask_dir + "/" + masks)
        #图像灰度化
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) 
        # cv2.imshow('masks', mask)
        # cv2.waitKey(0)
        cut_ROI(mask)
    