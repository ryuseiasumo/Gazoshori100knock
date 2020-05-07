import cv2
import numpy as np
import sys
sys.path.append("..")
from Question_01_10.q2 import Gray


def Biliner(img, magnification):
    H, W  = img.shape
    new_H = int(H * magnification)
    new_W = int(W * magnification)

    y = np.arange(new_H).repeat(new_W).reshape((new_H,-1))
    x = np.tile(np.arange(new_W), (new_H,1))

    ty = y/magnification
    tx = x/magnification

    iy = np.floor(ty).astype(np.uint8)
    ix = np.floor(tx).astype(np.uint8)

    ix = np.minimum(ix, W-2)
    iy = np.minimum(iy, H-2)

    dy = ty - iy
    dx = tx - ix

    out = (1-dx) * (1-dy) * img[iy, ix] + dx * (1 - dy) * img[iy, ix+1] + (1 - dx) * dy * img[iy+1, ix] + dx * dy * img[iy+1, ix+1]

    out = out.clip(0,255)
    out = out.astype(np.uint8)
    return out

img = cv2.imread("./image_71_80/imori.jpg").astype(np.float)
img_gray = Gray(img)
img_05 = Biliner(img_gray, 0.5)
img_ans = Biliner(img_05, 2)

cv2.imwrite("./image_71_80/answer_73.jpg",img_ans)
cv2.imshow("result", img_ans)
cv2.waitKey(0)
cv2.destroyAllWindows()
