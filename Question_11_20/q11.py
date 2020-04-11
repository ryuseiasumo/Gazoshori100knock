import cv2
import numpy as np

def zero_padding(_img, K_size = 3):
    pad = K_size//2
    img = _img.copy()
    out = np.pad(img, [(pad,pad),(pad,pad),(0,0)], "constant")
    return out

def mean_filter(_img, K_size = 3):
    img = _img.copy().astype(np.float32)
    out = np.zeros_like(img)
    pad = K_size//2

    if len(img.shape) == 3:
        H, W, C = img.shape

    else:
        img = np.expand_dims(img, axis = -1)
        H, W, C = img.shape

    H -= pad*2
    W -= pad*2

    for y in range(H):
        for x in range(W):
            for c in range(C):
                out[pad+y, pad+x, c] = np.mean(img[y:y+pad*2+1,x:x+pad*2+1, c])

    out = out.astype(np.uint8)
    return out[pad:H+pad,pad:W+pad]



img = cv2.imread("./image_11_20/imori.jpg")

img_zero = zero_padding(img)
img_ans = mean_filter(img_zero)


cv2.imwrite("./image_11_20/answer11.jpg", img_ans)
cv2.imshow("result", img_ans)
cv2.waitKey(0)
cv2.destroyAllWindows()
