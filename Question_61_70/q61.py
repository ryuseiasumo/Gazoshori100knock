import cv2
import numpy as np

def zero_padding(_img, K_size = 3):
    pad = K_size//2
    out = np.pad(_img,[(pad, pad),(pad, pad),(0,0)], "constant")
    return out

def four_connection(_img, K_size = 3):
    pad = K_size//2
    img = _img.copy().astype(np.float)
    if len(_img.shape) == 3:
        H, W, C = _img.shape
    else:
        _img = np.expand_dims(_img, axis = -1)
        H, W, C = _img.shape

    img_zero = zero_padding(img)

    #print(img_zero)
    tmp = np.zeros((H+2,W+2), dtype=np.int)

    # binarize
    tmp[img_zero[..., 0] > 0] = 1
    out = np.zeros_like(img_zero)

    for y in range(pad,H+pad):
        for x in range(pad,W+pad):
            if tmp[y,x] < 1:
                continue

            x1 = tmp[y, x+1]
            x2 = tmp[y-1, x+1]
            x3 = tmp[y-1, x]
            x4 = tmp[y-1, x-1]
            x5 = tmp[y, x-1]
            x6 = tmp[y+1, x-1]
            x7 = tmp[y+1, x]
            x8 = tmp[y+1, x+1]

            S =  (x1 - x1*x2*x3) + (x3 - x3*x4*x5) + (x5 - x5*x6*x7) + (x7 - x7*x8*x1)

            if S == 0:
                out[y,x] = [0,0,255]

            elif S == 1:
                out[y,x] = [0,255,0]

            elif S == 2:
                out[y,x] = [255,0,0]

            elif S == 3:
                out[y,x] = [255,255,0]

            elif S == 4:
                out[y,x] = [255, 0, 255]
                
    out = out[pad:H+pad, pad:W+pad]
    print(out)
    out = out.astype(np.uint8)
    return out

img = cv2.imread("./image_61_70/renketsu.png")

img_ans = four_connection(img)

cv2.imwrite("./image_61_70/answer_61.jpg", img_ans)
cv2.imshow("result", img_ans)
cv2.waitKey(0)
cv2.destroyAllWindows()
