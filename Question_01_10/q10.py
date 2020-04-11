import cv2
import numpy as np

def medianfilter(_img, K_size = 3):
    img = _img.copy()
    out = _img.copy()
    pad = K_size//2

    if len(img.shape) == 3:
        H, W, C = img.shape

    else:
        img = np.expand_dims(img, axis = -1)
        H, W, C = img.shape

    H -= pad*2
    W -= pad*2

    # median
    for y in range(H):
        for x in range(W):
            for c in range(C):
                out[y+pad, x+pad, c] = np.median(img[y:y+2*pad+1, x:x+2*pad+1, c])
    out = out.astype(np.uint8)
    print(out)
    print(out.shape)
    print("ーーーーーーーーーーーーーーーーーーーーーーー")
    return out[pad:H+pad, pad:W+pad]

def zero_padding(_img,K_size = 3):
    img = _img.copy()
    pad = K_size//2
    out = np.pad(img, [(pad,pad),(pad,pad),(0,0)],"constant")
    return out

img = cv2.imread("./image_01_10/imori_noise.jpg").astype(np.float32)
print(img)
print(img.shape)
print("ーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーー")
img_zero = zero_padding(img)
img_ans = medianfilter(img_zero)

print(img_ans)
print(img_ans.shape)

cv2.imwrite("./image_01_10/answer10.jpg", img_ans)
cv2.imshow("result", img_ans)
cv2.waitKey(0)
cv2.destroyAllWindows()
