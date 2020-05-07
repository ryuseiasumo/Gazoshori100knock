import cv2
import numpy as np

def affine(_img, a, b, c, d, tx = 30, ty = -30):
    H, W, C = _img.shape  #(128,128,3)

    #ゼロパディングと同様→(0,x),(129,x),(y,0),(y,129)の画素値は0！→本来の画像から外れた座標(-31,100)などを、下でnp.minimumやnp.maxmunによって,(0,100)などに合わせる→画素値0が代入される って仕様にするため。
    img = np.zeros((H+2, W+2, C), dtype = np.float)  #(130,130,3)
    img[1:H+1, 1:W+1] = _img
    # print(img)
    # print("ーーーーーーーーーーーーーーーーーー")

    H_new = np.round(H*d).astype(np.int)
    W_new = np.round(W*a).astype(np.int)
    out = np.zeros((H_new+1,W_new+1,C),dtype= np.float)  #(129,129,3) a=1,d=1の時

    # print(out)
    # print("ーーーーーーーーーーーーーーーーーー")
    # print(out.shape)

    # get position of new image
    y_new = np.arange(H_new).repeat(W_new).reshape(H_new,-1)
    x_new = np.tile(np.arange(W_new),(H_new,1))
    print(y_new)
    print(x_new)

    # get position of original image by affine（逆行列を使って元画像の対応する座標を見つける）
    adbc = a*d - b*c
    y = np.round((-c*x_new+a*y_new)/adbc).astype(np.int) - ty + 1
    x = np.round((d*x_new -b*y_new)/adbc).astype(np.int) - tx + 1   #1足す理由は、imgにパディングを施しているため！

    # print(y)
    # print(x)

    # 0より小さい座標、W+1より大きい座標を消す。
    x = np.minimum(np.maximum(x, 0), W+1).astype(np.int)
    y = np.minimum(np.maximum(y, 0), H+1).astype(np.int)

    print(y)
    print(x)

    # assgin pixcel to new image
    out[y_new, x_new] = img[y, x]

    print(out)
    out = out[:H_new, :W_new]
    out = out.astype(np.uint8)

    print(out)

    return out





img = cv2.imread("./image_21_30/imori.jpg").astype(np.float)
img_ans = affine(img, 1, 0, 0, 1)
cv2.imwrite("./image_21_30/answer28.jpg", img_ans)
cv2.imshow("result", img_ans)
cv2.waitKey(0)
cv2.destroyAllWindows()
