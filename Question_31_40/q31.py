import cv2
import numpy as np

def affine(_img, a, b, c, d, tx, ty):
    H, W, C = _img.shape
    img = np.zeros((H+2, W+2 ,C))
    img[1:H+1,1:W+1] = _img.copy()

    #new_scale
    new_H = int(d*H + c*W)
    new_W = int(a*W + b*H)
    out = np.zeros((new_H+1, new_W+1 ,C))

    new_y = np.arange(new_H).repeat(new_W).reshape(new_H,-1)
    new_x = np.tile(np.arange(new_W),(new_H,1))

    adbc = a*d - b*c
    y = np.round((-c*new_x + d*new_y)/adbc).astype(np.int) - ty + 1
    x = np.round((a*new_x - b*new_y)/adbc).astype(np.int) - tx + 1

    y = np.minimum(np.maximum(y, 0), H+1).astype(np.int)
    x = np.minimum(np.maximum(x, 0), W+1).astype(np.int)

    print(img.shape)
    print(out.shape)
    out[new_y,new_x] = img[y,x]

    out = out[:new_H,:new_W].astype(np.uint8)
    return out


img = cv2.imread("./image_31_40/imori.jpg").astype(np.float)
H ,W, C = img.shape
dx = 30
dy = 30
x_img = affine(img , 1, dx/H, 0, 1, 0 ,0)
y_img = affine(img , 1, 0, dy/W, 1, 0 ,0)
xy_img = affine(img, 1, dx/H, dy/W, 1, 0 ,0)

cv2.imwrite("./image_31_40/answer31_1.jpg",x_img)
cv2.imshow("result", x_img)
cv2.waitKey(0)
cv2.destroyWindow("result")

cv2.imwrite("./image_31_40/answer31_2.jpg",y_img)
cv2.imshow("result_2", y_img)
cv2.waitKey(0)
cv2.destroyWindow("result_2")

cv2.imwrite("./image_31_40/answer31_3.jpg",xy_img)
cv2.imshow("result_3", xy_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
