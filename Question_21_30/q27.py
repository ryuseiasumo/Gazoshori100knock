import cv2
import numpy as np


# def weight(t):
#     a  = -1
#     if np.abs(t) <= 1:
#         return (a+2)*np.abs(t)**3 - (a+3)*np.abs(t)**2 + 1
#     elif 1 < np.abs(t) <= 2:
#         return a*np.abs(t)**3 - 5*a*np.abs(t)**2 + 8*a*np.abs(t) -4*a
#     else:
#         return 0

def weight(t):
    a = -1.
    at = np.abs(t)
    w = np.zeros_like(t)
    ind = np.where(at <= 1)
    w[ind] = ((a+2) * np.power(at, 3) - (a+3) * np.power(at, 2) + 1)[ind]
    ind = np.where((at > 1) & (at <= 2))
    w[ind] = (a*np.power(at, 3) - 5*a*np.power(at, 2) + 8*a*at - 4*a)[ind]
    return w

def Bi_cubic(_img, ay = 1.5, ax = 1.5):
    img = _img.copy()
    H, W, C = img.shape

    aH = int(ay * H)
    aW = int(ax * W)

    y = np.arange(aH).repeat(aW).reshape(aH,-1)
    x = np.tile(np.arange(aW), (aH,1))

    y = y/ay
    x = x/ax

    iy = np.floor(y).astype(np.uint8)
    ix = np.floor(x).astype(np.uint8)

    iy = np.minimum(iy, H-1) #127だけ、126になる
    ix = np.minimum(ix, W-1)


    print(iy,ix)
    print("ーーーーーーーーーーーーーーー")
    print(y,x)
    print("ーーーーーーーーーーーーーーー")

    dy2 = abs(iy - y)
    dx2 = abs(ix - x)
    dy1 = dy2 + 1
    dx1 = dx2 + 1
    dy3 = abs(dy2-1)
    dx3 = abs(dx2-1)
    dy4 = abs(dy2-2)
    dx4 = abs(dy2-2)

    dys = [dy1, dy2, dy3, dy4]
    dxs = [dx1, dx2, dx3, dx4]

    w_sum = np.zeros((aH, aW, C), dtype=np.float32)
    out = np.zeros((aH, aW, C), dtype=np.float32)

    # interpolate
    for j in range(-1, 3):
        for i in range(-1, 3):
            # ind_x = ix + i   #127をこえてしまうやつがある
            # ind_y = iy + j
            # print(ind_x[ind_x>127])
            # print(ind_y[ind_y>127])

            ind_x = np.minimum(np.maximum(ix + i, 0), W-1)  #画素の存在する範囲をこえないようにしている(端のほうは、その特徴が強く表れる？)
            ind_y = np.minimum(np.maximum(iy + j, 0), H-1)
            print(ind_x)
            print(ind_y)
            wx = weight(dxs[i+1])
            wy = weight(dys[j+1])
            print("ーーーーーーーーーーー")
            print(wx)
            wx = np.repeat(np.expand_dims(wx, axis=-1), 3, axis=-1) #3色あるため
            wy = np.repeat(np.expand_dims(wy, axis=-1), 3, axis=-1)

            w_sum += wx * wy
            out += wx * wy * img[ind_y, ind_x]

    out /= w_sum
    out = np.clip(out, 0, 255)
    out = out.astype(np.uint8)

    return out



img = cv2.imread("./image_21_30/imori.jpg").astype(np.float)
img_ans = Bi_cubic(img)
cv2.imwrite("./image_21_30/answer27.jpg", img_ans)
cv2.imshow("result", img_ans)
cv2.waitKey(0)
cv2.destroyAllWindows()
