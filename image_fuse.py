###################################注意事项######################################
# cv2读取图片的宽高和其他函数要求的顺序可能是不同的
# cv2.imread 是(x, y), poly 是(y,x) !!!!x 是垂直方向， y 是水平方向!!!!!!
##################################################################################

# Standard imports
import cv2
import numpy as np 


def get_rois(src, dst, poly_from_json, center, resize_num = 1):
    # Create a rough mask around the src.
    src_mask = np.zeros(src.shape, src.dtype)
    # Resize
    # dst = cv2.resize(dst, (500, 300), interpolation = cv2.INTER_CUBIC)

    ## 生成mask
    poly = np.array(poly_from_json,  np.int32)
    cv2.fillPoly(src_mask, [poly], (255, 255, 255)) #原图的mask



    ##  获取包裹mask最小方框的信息
    y_max, y_min, x_max, x_min = np.max(poly[:, 0]), np.min(poly[:, 0]), np.max(poly[:, 1]), np.min(poly[:, 1]) # mask最小方框的坐标
    width = y_max - y_min # 最小方框的宽
    height =  x_max - x_min # 最小方框的高 

    ## 获取src,dst和mask中的对应的方框
    src_roi = src[int(x_min): int(x_max), int(y_min): int(y_max)] # src 中的roi
    mask_roi = src_mask[int(x_min): int(x_max), int(y_min): int(y_max)] # src_mask 中的roi
    box = [int(center[1] - height / 2), int(center[1] + height / 2), int(center[0] - width / 2), int(center[0] + width / 2)]
    dst_roi = dst[box[0]:box[1], box[2]:box[3]] # dst 中的roi

    if resize_num != 1:
        # resize mask
        mask_roi = cv2.resize(mask_roi, None, fx=resize_num, fy=resize_num, interpolation = cv2.INTER_AREA)

        # get new src_roi corresponding the mask_roi
        height, width = mask_roi.shape[:2]
        new_y_max, new_y_min = int((y_max + y_min) / 2 + width / 2), int((y_max + y_min) / 2 - width / 2)
        new_x_max, new_x_min = int((x_max + x_min) / 2 + height / 2), int((x_max + x_min) / 2 - height / 2) 

        src_roi = src[int(new_x_min): int(new_x_max), int(new_y_min): int(new_y_max)]

        # get new box and dst_roi
        box = [int(center[1] - height / 2 ), int(center[1] + height / 2), int(center[0] - width / 2), int(center[0] + width / 2)]
        dst_roi = dst[box[0]:box[1], box[2]:box[3]]

    return [src_roi, dst_roi, mask_roi], box

def image_fusion(dst, box, alpha, rois = []):
    if rois != []:
        ## 按位操作，获取
        img2gray = cv2.cvtColor(rois[2],cv2.COLOR_BGR2GRAY)      # 将图片灰度化,变成单通道
        ret, mask = cv2.threshold(img2gray, 175, 255, cv2.THRESH_BINARY)#ret是阈值（175）mask是二值化图像， 虽然已经是二值化的图片了，但是没有这步效果很差，不知道为啥 
        mask_inv = cv2.bitwise_not(mask)#获取把logo的区域取反按位运算


        src_smoke_only  =  cv2.bitwise_and(rois[0], rois[0], mask = mask) # 截取烟雾，非截取部分取值为0
        dst_smoke_only  =  cv2.bitwise_and(rois[1], rois[1], mask = mask) # 截取烟雾在背景区域的部分，非截取部分取值为0
        dst_bg_only  =  cv2.bitwise_and(rois[1],rois[1],mask = mask_inv) # 截取背景
        new_smoke= cv2.addWeighted(src_smoke_only, alpha, dst_smoke_only, 1-alpha, 0)
        add_box = cv2.add(new_smoke, dst_bg_only)#相加即可
        add_box = cv2.blur(add_box,(3,3))

        add_whole =  dst.copy()
        add_whole[box[0]:box[1], box[2]:box[3]] = add_box

        return add_box, add_whole



###### input  ################
# poly from json
# src_path, dst_path
#

#  poly from json
poly_from_json = [ [
          396.2770562770563,
          160.6060606060606
        ],
        [
          397.5757575757575,
          170.56277056277057
        ],
        [
          401.9047619047619,
          175.75757575757575
        ],
        [
          412.72727272727275,
          185.28138528138527
        ],
        [
          420.95238095238096,
          195.67099567099567
        ],
        [
          438.7012987012987,
          183.54978354978354
        ],
        [
          443.030303030303,
          167.09956709956708
        ],
        [
          444.7619047619047,
          154.978354978355
        ],
        [
          421.81818181818176,
          148.91774891774892
        ],
        [
          404.069264069264,
          153.67965367965368
        ] ]

# Read images
src_path = "images/smoke.jpg"
dst_path = "images/background.jpg"
src = cv2.imread(src_path)
dst = cv2.imread(dst_path)

# 这是 CENTER 所在的地方
center = [int(0.5 * dst.shape[1]), int(0.5 * dst.shape[0])]  # center 是(y, x)-(width, height) cv2.imread 是(x, y), poly 是(y,x) x 是垂直方向-height， y 是水平方向-width

alpha = 0.3
# rois, box= get_rois(src, dst, poly_from_json, center)
# add_box, fusion = image_fusion(dst, box, rois)


# rois, box= get_rois(src, dst, poly_from_json, center, resize_num = 0.5)
rois, box= get_rois(src, dst, poly_from_json, center)
add_box, fusion = image_fusion(dst, box, alpha, rois)      
height, width = (box[1] - box[0]), (box[3] - box[2])
height_range, width_range = int(int(dst.shape[0]) - height - 1), int(int(dst.shape[1]) - width - 1)


def callback(object):
    pass

cv2.namedWindow('image', cv2.WINDOW_KEEPRATIO) # 窗口大小保持比例

cv2.createTrackbar('height', 'image', int(height_range / 2) , height_range, callback)
cv2.createTrackbar('width', 'image', int(width_range / 2), width_range, callback)
cv2.createTrackbar('alpha', 'image', 70 , 100, callback)
cv2.createTrackbar('resize', 'image', 70 , 100, callback)


n = 1
while True:
    center[1] = int(cv2.getTrackbarPos('height', 'image')) + int(height / 2)
    center[0] = int(cv2.getTrackbarPos('width', 'image')) + int(width / 2 + 1)
    alpha = cv2.getTrackbarPos('alpha', 'image') / 100
    resize_num = cv2.getTrackbarPos('resize', 'image') / 100
    rois, box= get_rois(src, dst, poly_from_json, center, resize_num)
    add_box, fusion = image_fusion(dst, box, alpha, rois)
    
    cv2.imshow('image', fusion)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

    if cv2.waitKey(10) & 0xFF == ord('s'):
        cv2.imwrite(dst_path.replace('.jpg', '')+ '_fusion_' + str(n) +'.jpg', fusion)
        n += 1


cv2.destroyAllWindows()

            