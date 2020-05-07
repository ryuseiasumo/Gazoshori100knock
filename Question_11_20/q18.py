import cv2
import sys
sys.path.append("..")
import numpy as np
from Question_01_10.q2 import Gray

def zero_padding(_img, K_size =3 ):
    pad = K_size//2
    out = np.pad(_img, ([pad,pad],[pad,pad]), "constant")
    return out

def emboss_fillter(_img, K_size = 3):
    img = _img.copy()
    pad = K_size//2

    if len(_img.shape) == 3:
        H, W, C = _img.shape
    else:
        _img = np.expand_dims(_img, axis = -1)
        H, W, C = _img.shape

    img_zero = zero_padding(img).astype(np.float)
    out = img_zero.copy()

    K = [[-2., -1., 0.],[-1., 1., 1.],[0., 1., 2.]]

    for y in range(H):
        for x in range(W):
            out[pad+y, pad+x] = np.sum(K*img_zero[y:y+K_size, x:x+K_size])

    print(out)
    print("ーーーーーーーーーーーーーー")

    out = np.clip(out, 0, 255)

    print(out)
    print("ーーーーーーーーーーーーーー")

    out = out[pad:pad+H, pad:pad+W].astype(np.uint8)
    return out



img = cv2.imread("./image_11_20/imori.jpg").astype(np.float)
img_gray = Gray(img)
img_ans = emboss_fillter(img_gray)

cv2.imwrite("./image_11_20/answer18.jpg", img_ans)
cv2.imshow("result", img_ans)
cv2.waitKey(0)
cv2.destroyAllWindows()
