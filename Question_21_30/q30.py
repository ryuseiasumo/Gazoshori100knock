import cv2
import numpy as np

def affine(_img, a, b, c, d, tx, ty, t):
    H, W, C = _img.shape

    #zero_パディング
    img = np.zeros((H+2, W+2, C),dtype = np.float)
    img[1:H+1, 1:W+1] = _img.copy()

    #outのスケール
    new_H = np.round(H).astype(np.int)
    new_W = np.round(W).astype(np.int)
    out = np.zeros((new_H+1, new_W+1, C),dtype = np.float)

    new_y = np.arange(new_H).repeat(new_W).reshape(new_H,-1)
    new_x = np.tile(np.arange(new_W), (new_H,1))

    adbc = a*d-b*c
    y = np.round(-c*new_x + d*new_y/adbc).astype(np.int) - ty + 1
    x = np.round( a*new_x - b*new_y/adbc).astype(np.int) - tx + 1

    # y = np.minimum(np.maximum(y, 0), H+1).astype(np.int)
    # x = np.minimum(np.maximum(x, 0), W+1).astype(np.int)

    # adjust center by affine
    if t == 0:
        print("-----------------")
        print(x.max())
        print("-----------------")
        print(x.min())
        dcx = (x.max() + x.min()) // 2 - W // 2
        dcy = (y.max() + y.min()) // 2 - H // 2

        x -= dcx
        y -= dcy

        print("-----------------")
        print(dcx)
        print("-----------------")
        print(dcy)
        print("-----------------")

    print(x)
    print("-----------------")
    print(y)
    print("-----------------")
    x = np.clip(x, 0, W + 1)
    y = np.clip(y, 0, H + 1)

    out[new_y,new_x] = img[y,x]

    out = out[:new_H,:new_W].astype(np.uint8)
    return out

img = cv2.imread("./image_21_30/imori.jpg").astype(np.float)
# Affine
A = 30.
theta = - np.pi * A / 180.

img_ans_1 = affine(img, np.cos(theta), -np.sin(theta), np.sin(theta),np.cos(theta),0,0,1)
img_ans_2 = affine(img, np.cos(theta), -np.sin(theta), np.sin(theta),np.cos(theta),0,0,0)

cv2.imwrite("./image_21_30/answer30_1.jpg", img_ans_1)
cv2.imshow("result", img_ans_1)
cv2.waitKey(0)
cv2.destroyWindow("result")

cv2.imwrite("./image_21_30/answer30_2.jpg", img_ans_2)
cv2.imshow("result_2", img_ans_2)
cv2.waitKey(0)
cv2.destroyAllWindows()
