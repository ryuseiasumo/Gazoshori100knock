import cv2
import numpy as np

def affine(_img, a, b ,c, d, tx, ty):
    H, W, C = _img.shape

    #zero_パディング
    img = np.zeros((H+2, W+2, C),dtype = np.float)
    img[1:H+1, 1:W+1] = _img.copy()

    #outのスケール
    new_H = np.round(H*d).astype(np.int)
    new_W = np.round(W*a).astype(np.int)
    out = np.zeros((new_H+1, new_W+1, C),dtype = np.float)  #1足している理由は？余裕もってるのか？？

    # get position of new image
    new_y = np.arange(new_H).repeat(new_W).reshape(new_H,-1)
    new_x = np.tile(np.arange(new_W), (new_H,1))
    print(new_y.shape)
    print("ーーーーーーーーーーー")
    print(new_x.shape)
    print("ーーーーーーーーーーー")

    # print(len(new_y),len(new_x))
    #アフィンの逆行列を用いて、元の画像でポジショニング
    adbc = a*d-b*c
    y = np.round((-c*new_x+a*new_y)/adbc).astype(np.int) - ty +1
    x = np.round((-b*new_y+d*new_x)/adbc).astype(np.int) - tx +1
    print(y)
    print("ーーーーーーーーーーー")
    print(x)
    print("ーーーーーーーーーーー")

    # 0より小さい座標、W+1(129)より大きい座標を消す。元画像の画素値が存在しているのは、(1,1)~(H,W)！
    y = np.minimum(np.maximum(y, 0), H+1).astype(np.int)
    x = np.minimum(np.maximum(x, 0), W+1).astype(np.int)
    print(y.shape)
    print(x.shape)

    print(y)
    print("ーーーーーーーーーーー")
    print(x)
    print("ーーーーーーーーーーー")
    # print(len(y),len(x))

    out[new_y,new_x] = img[y,x]
    print(out[new_H,new_W])
    out = out[:new_H, :new_W].astype(np.uint8)
    print("ーーーーーーーーーーー")
    print(out)
    return out


img = cv2.imread("./image_21_30/imori.jpg").astype(np.float)
img_ans_1 = affine(img, 1.3, 0, 0, 0.8, 0, 0)
img_ans_2 = affine(img, 1.3, 0, 0, 0.8, 30, -30)

cv2.imwrite("./image_21_30/answer29_1.jpg", img_ans_1)
cv2.imshow("result", img_ans_1)
cv2.waitKey(0)
cv2.destroyWindow("result")

cv2.imwrite("./image_21_30/answer29_2.jpg", img_ans_2)
cv2.imshow("result_2", img_ans_2)
cv2.waitKey(0)
cv2.destroyAllWindows()
