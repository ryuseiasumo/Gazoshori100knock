import cv2
import numpy as np

def Bi_linear(_img, ay = 1.5, ax= 1.5):
    img = _img.copy()
    H, W, C = img.shape

    aH = int(ay * H)
    aW = int(ax * W)

    y = np.arange(aH).repeat(aW).reshape(aH,-1)
    x = np.tile(np.arange(aW), (aH, 1))

    # print(y)
    # print(x)

    y = (y / ay)
    x = (x / ax)

    ix = np.floor(x).astype(np.int)
    iy = np.floor(y).astype(np.int)

    iy = np.minimum(iy, H-2) #127だけ126になる
    ix = np.minimum(ix, W-2)
    # print(iy)
    # print(ix)

    #距離を求める
    dy = y - iy
    dx = x - ix

    #二次元配列を三次元配列に変換(色成分を追加)
    dy = np.repeat(np.expand_dims(dy, axis = -1), 3, axis = -1)
    dx = np.repeat(np.expand_dims(dx, axis = -1), 3, axis = -1)

    print(dy, dx)

    # interpolation
    out = (1-dx) * (1-dy) * img[iy, ix] + dx * (1 - dy) * img[iy, ix+1] + (1 - dx) * dy * img[iy+1, ix] + dx * dy * img[iy+1, ix+1]
    out.clip(0,255)
    out = out.astype(np.uint8)

    print(out)
    return out




img = cv2.imread("./image_21_30/imori.jpg")
img_ans = Bi_linear(img)

cv2.imwrite("./image_21_30/answer26.jpg", img_ans)
cv2.imshow("result", img_ans)
cv2.waitKey(0)
cv2.destroyAllWindows()
