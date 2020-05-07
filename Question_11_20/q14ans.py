import sys
sys.path.append("..")
import cv2
import numpy as np
from Question_01_10.q2 import Gray

def zero_padding(_img, K_size = 3):
    img = _img.copy()
    pad = K_size//2
    out = np.pad(img, [(pad,pad), (pad,pad), (0,0)], "constant")
    return out

def differentialfilter(_img, K_size = 3):
    img = _img.copy()
    pad = K_size//2
    if len(img.shape) == 3:
        H,W,C = img.shape
    else:
        img = np.expand_dims(img ,axis = -1)
        H, W, C = img.shape

    img_zero = zero_padding(img)
    out = np.zeros_like(img_zero).astype(np.float)
    tmp = img_zero.copy().astype(np.float)

    out_v = out.copy()
    out_h = out.copy()

    #prepare Kernel
    Kernel_v = [[[0.], [-1.], [0.]], [[0.], [1.], [0.]], [[0.], [0.], [0.]]]
    Kernel_h = [[[0.], [0.], [0.]], [[-1.], [1.], [0.]], [[0.], [0.], [0.]]]

    for y in range(H):
        for x in range(W):
            out_v[pad+y,pad+x] = np.sum(Kernel_v* (tmp[y: y + K_size, x: x + K_size]))
            out_h[pad+y,pad+x] = np.sum(Kernel_h*(tmp[y: y + K_size, x: x + K_size]))


    out_v = np.clip(out_v, 0, 255)
    out_h = np.clip(out_h, 0, 255)
    out_v = out_v[pad:pad+H,pad:pad+W].astype(np.uint8)
    out_h = out_h[pad:pad+H,pad:pad+W].astype(np.uint8)

    print(out_v)
    print(out_h)
    return out_v, out_h



img = cv2.imread("./image_11_20/imori.jpg").astype(np.float)
img_gray = Gray(img)
img_ans_v, img_ans_h = differentialfilter(img_gray)

cv2.imwrite("./image_11_20/answer14_v.jpg", img_ans_v)
cv2.imshow("result_v", img_ans_v)
while cv2.waitKey(100) != 27:# loop if not get ESC
    if cv2.getWindowProperty('result_v',cv2.WND_PROP_VISIBLE) <= 0:
        break
cv2.destroyWindow('result_v')


cv2.imwrite("./image_11_20/answer14_h.jpg", img_ans_h)
cv2.imshow("result_h", img_ans_h)
while cv2.waitKey(100) != 27:
    if cv2.getWindowProperty('result_h',cv2.WND_PROP_VISIBLE) <= 0:
        break
cv2.destroyWindow('result_h')
cv2.destroyAllWindows()
