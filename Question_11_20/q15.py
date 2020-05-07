import cv2
import sys
sys.path.append("..")
import numpy as np
from Question_01_10.q2 import Gray

def zero_padding(_img, K_size = 3):
    pad = K_size//2
    out = np.pad(_img,[(pad, pad),(pad, pad)], "constant")
    return out

def prewitt(_img, K_size = 3):
    pad = K_size//2
    img = _img.copy().astype(np.float)
    if len(_img.shape) == 3:
        H, W, C = _img.shape
    else:
        _img = np.expand_dims(_img, axis = -1)
        H, W, C = _img.shape

    img_zero = zero_padding(img)
    out = img_zero.copy().astype(np.float)
    out_v = out.copy()
    out_h = out.copy()

    #prepare Kernel
    K_v = [[1.,1.,1.],[0.,0.,0.],[-1.,-1.,-1.]]
    K_h = [[-1.,0.,1.],[-1.,0.,1.],[-1.,0.,1.]]
    for y in range(H):
        for x in range(W):
            out_v[pad+y, pad+x] = np.sum(K_v * out[y:y+K_size, x:x+K_size])
            out_h[pad+y, pad+x] = np.sum(K_h * out[y:y+K_size, x:x+K_size])

    print(out_v)
    print("ーーーーーーーーーーーーーー")
    print(out_h)
    print("ーーーーーーーーーーーーーー")

    out_v = np.clip(out_v, 0, 255)
    out_h = np.clip(out_h, 0, 255)

    print(out_v)
    print("ーーーーーーーーーーーーーー")
    print(out_h)
    print("ーーーーーーーーーーーーーー")
    out_v = out_v[pad:pad+H, pad:pad+W].astype(np.uint8)
    out_h = out_h[pad:pad+H, pad:pad+W].astype(np.uint8)
    return out_v, out_h


img = cv2.imread("./image_11_20/imori.jpg")
img_gray = Gray(img)
img_ans_v, img_ans_h = prewitt(img_gray)

cv2.imwrite("./image_11_20/answer15_v.jpg", img_ans_v)
cv2.imwrite("./image_11_20/answer15_h.jpg", img_ans_h)
cv2.imshow("result_v", img_ans_v)
cv2.waitKey(0)
cv2.destroyWindow("result_v")

cv2.imshow("result_h", img_ans_h)
cv2.waitKey(0)
cv2.destroyAllWindows()
