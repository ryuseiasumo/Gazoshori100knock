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


def max_min(_img, K_size = 3):
    img = _img.copy()
    pad = K_size//2
    if len(img.shape) == 3:
        H, W, C = img.shape
    else:
        img = np.expand_dims( img, axis = -1)
        H, W, C = img.shape

    img_zero = zero_padding(img).astype(np.float32)

    out = np.zeros_like(img)
    max_img = np.zeros_like(img)
    min_img = np.zeros_like(img)

    for y in range(H):
        for x in range(W):
            for c in range(C):
                max_img[y,x,c] = np.max(img_zero[y:y+2*pad+1, x:x+2*pad+1, c])
                min_img[y,x,c] = np.min(img_zero[y:y+2*pad+1, x:x+2*pad+1, c])

    out = max_img-min_img
    return out

img = cv2.imread("./image_11_20/imori.jpg")
img_gray = Gray(img)
img_ans = max_min(img_gray)
img_ans = img_ans.astype(np.uint8)
print(img_ans)

cv2.imwrite("./image_11_20/answer13.jpg", img_ans)
cv2.imshow("result", img_ans)
cv2.waitKey(0)
cv2.destroyAllWindows()
