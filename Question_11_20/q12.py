import cv2
import numpy as np

def motion_fillter(_img, K_size = 3):
    img = _img.copy()
    pad = K_size//2

    if len(img.shape) == 3:
        H, W, C = img.shape

    else:
        img = np.expand_dims(img, axis = -1)
        H, W, C = img.shape

    img = zero_padding(img)
    out = img.copy().astype(np.float32)

    #prepare Kernel
    Kernel = np.diag([1]*K_size).astype(np.float32)
    Kernel /= K_size

    for y in range(H):
        for x in range(W):
            for c in range(C):
                out[pad+y, pad+x, c] = np.sum(Kernel*out[y:y+2*pad+1,x:x+2*pad+1,c])

    out = out.astype(np.uint8)
    return out[pad:H+pad,pad:W+pad]


def zero_padding(_img, K_size = 3):
    pad = K_size//2
    img = _img.copy()
    out = np.pad(img, [(pad,pad), (pad, pad), (0,0)], "constant")
    return out

img = cv2.imread("./image_11_20/imori.jpg")

img_ans = motion_fillter(img)

cv2.imwrite("./image_11_20/answer12.jpg", img_ans)
cv2.imshow("result", img_ans)
cv2.waitKey(0)
cv2.destroyAllWindows()
