import cv2
import sys
sys.path.append("..")
import numpy as np
from Question_01_10.q2 import Gray

def zero_padding(_img, K_size = 5):
    pad = K_size//2
    out = np.pad(_img, ([pad,pad],[pad,pad]), "constant")
    return out

def Log_fillter(_img, K_size = 5, sigma = 3):
    img = _img.copy()
    pad = K_size//2

    if len(_img.shape) == 3:
        H, W, C = _img.shape
    else:
        _img = np.expand_dims(_img, axis = -1)
        H, W, C = _img.shape

    img_zero = zero_padding(img).astype(np.float)
    tmp = img_zero.copy()
    out = np.zeros_like(tmp).astype(np.float)

    print(tmp)
    print("ーーーーーーーーー")

    #prepare kernel
    K = np.zeros((K_size, K_size), dtype = np.float)
    for y in range(-pad, -pad + K_size):
        for x in range(-pad, -pad + K_size):
            K[y + pad, x + pad] = (x ** 2 + y ** 2 - 2 * (sigma ** 2)) * np.exp( - (x ** 2 + y ** 2) / (2 * (sigma ** 2)))
    K /= (2 * np.pi * (sigma ** 6))
    K /= K.sum()

    # filtering
    for y in range(H):
        for x in range(W):
            out[pad + y, pad + x] = np.sum(K * tmp[y: y + K_size, x: x + K_size])

    print(out)
    print("ーーーーーーーーーーーーーー")

    out = np.clip(out, 0, 255)

    print(out)
    print("ーーーーーーーーーーーーーー")

    out = out[pad:pad+H, pad:pad+W].astype(np.uint8)
    return out



img = cv2.imread("./image_11_20/imori_noise.jpg").astype(np.float)
img_gray = Gray(img)
img_ans = Log_fillter(img_gray)

cv2.imwrite("./image_11_20/answer19.jpg", img_ans)
cv2.imshow("result", img_ans)
cv2.waitKey(0)
cv2.destroyAllWindows()
