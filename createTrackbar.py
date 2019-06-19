# -*- coding: utf-8 -*- 

import cv2

#两个回调函数
def GaussianBlurSize(GaussianBlur_size):
    print(GaussianBlur_size)
    global KSIZE 
    KSIZE = GaussianBlur_size * 2 +3
    # print KSIZE, SIGMA
    dst = cv2.GaussianBlur(scr, (KSIZE,KSIZE), SIGMA, KSIZE) 
    cv2.imshow(window_name,dst)

def GaussianBlurSigma(GaussianBlur_sigma):
    print(GaussianBlur_sigma)
    global SIGMA
    SIGMA = GaussianBlur_sigma/10.0
    # print KSIZE, SIGMA
    dst = cv2.GaussianBlur(scr, (KSIZE,KSIZE), SIGMA, KSIZE) 
    cv2.imshow(window_name,dst)

#全局变量
GaussianBlur_size = 1
GaussianBlur_sigma = 15

KSIZE = 1
SIGMA = 15
max_value = 300
max_type = 6
window_name = "GaussianBlurS Demo"
trackbar_size = "Size*2+3"
trackbar_sigema = "Sigma/10"

#读入图片，模式为灰度图，创建窗口
scr = cv2.imread("./images/background1.jpg",0)
cv2.namedWindow(window_name)

#创建滑动条
cv2.createTrackbar( trackbar_size, window_name, \
                    1, max_type, GaussianBlurSize ) #  GaussianBlurSize(GaussianBlur_size) = bar_value, 进度条上的值被传入recall的函数
cv2.createTrackbar( trackbar_sigema, window_name, \
                    15, max_value, GaussianBlurSigma )
#初始化
GaussianBlurSize(1)
GaussianBlurSigma(15)

if cv2.waitKey(0) == 27:  
    cv2.destroyAllWindows()
