# Standard imports
import cv2
import numpy as np 

# Read images
src = cv2.imread("images/DSC_0182_780.jpg")
dst = cv2.imread("images/dst_add_small.jpg")

# Create a rough mask around the airplane.
src_mask = np.zeros(src.shape, src.dtype)


# Resize
# dst = cv2.resize(dst, (500, 300), interpolation = cv2.INTER_CUBIC)
# crop_size_airplane =  (300, 194)
# img_new_airplane = cv2.resize(src, crop_size_airplane, interpolation = cv2.INTER_CUBIC)
# cv2.imwrite("images/resize_airplane.jpg", img_new_airplane);

# crop_size_sky = (1000, 561)
# img_new_sky = cv2.resize(dst, crop_size_sky, interpolation = cv2.INTER_CUBIC)
# cv2.imwrite("images/resize_sky.jpg", img_new_sky);


# 当然我们比较懒得话，就不需要下面两行，只是效果差一点。
# 不使用的话我们得将上面一行改为 mask = 255 * np.ones(obj.shape, obj.dtype) <-- 全白
poly_1 = np.array([   [
          951.9890109890109,
          512.0879120879121
        ],
        [
          934.4065934065934,
          486.8131868131868
        ],
        [
          920.1208791208792,
          476.9230769230769
        ],
        [
          901.4395604395604,
          478.021978021978
        ],
        [
          882.7582417582416,
          473.6263736263736
        ],
        [
          876.164835164835,
          459.34065934065933
        ],
        [
          886.0549450549449,
          431.86813186813185
        ],
        [
          895.9450549450548,
          415.38461538461536
        ],
        [
          913.5274725274726,
          394.5054945054945
        ],
        [
          942.098901098901,
          357.1428571428571
        ],
        [
          960.7802197802198,
          348.35164835164835
        ],
        [
          981.6593406593406,
          348.35164835164835
        ],
        [
          997.0439560439561,
          342.85714285714283
        ],
        [
          1027.8131868131868,
          335.16483516483515
        ],
        [
          1041.0,
          334.0659340659341
        ],
        [
          1057.4835164835165,
          346.15384615384613
        ],
        [
          1069.5714285714284,
          365.9340659340659
        ],
        [
          1062.9780219780218,
          397.8021978021978
        ],
        [
          1049.7912087912086,
          409.8901098901099
        ],
        [
          1025.6153846153845,
          428.57142857142856
        ],
        [
          995.9450549450548,
          427.4725274725275
        ],
        [
          967.3736263736264,
          435.16483516483515
        ],
        [
          957.4835164835165,
          447.2527472527472
        ],
        [
          953.087912087912,
          460.4395604395604
        ],
        [
          947.5934065934066,
          474.7252747252747
        ],
        [
          947.5934065934066,
          480.2197802197802
        ],
        [
          954.1868131868132,
          489.010989010989
        ] ], np.int32)



cv2.fillPoly(src_mask, [poly_1], (255, 255, 255))

# 这是 飞机 CENTER 所在的地方
center = (700, 400)

# Clone originally
# np.zeros(np.shape(img0), dtype=np.uint8)
# output_ORIGIN = cv2.add(src, dst, mask=src_mask)
# cv2.imwrite("images/original.jpg", output_ORIGIN)


# Clone seamlessly.
output_NORMAL = cv2.seamlessClone(src, dst, src_mask, center, cv2.NORMAL_CLONE)
output_MIXED = cv2.seamlessClone(src, dst, src_mask, center, cv2.MIXED_CLONE)
cv2.imwrite("images/normal.jpg", output_NORMAL)
cv2.imwrite("images/mixed.jpg", output_NORMAL)


      
