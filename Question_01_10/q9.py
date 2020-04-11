import cv2
import numpy as np

def zero_pad(_img):
    img = _img.copy()
    out = np.pad(img, [(1,1),(1,1),(0,0)], "constant")
    # print(out)
    return out


def gaussian(_img, K_size = 3, sigma = 1.3):
    img = _img.copy()
    if len(img.shape) == 3:
        H, W, C = img.shape

    else:  #グレースケールだった場合(すなわち、次元数が2の時)を想定？
        img = np.expand_dims(img, axis = -1) #末尾に次元を拡張
        H, W, C = img.shape

    H = H-2
    W = W-2

    out = img.copy()

    pad = K_size//2

    #prepare kernel
    Kernel = np.zeros((K_size, K_size), dtype = np.float32)
    for y in range(-pad, -pad + K_size):
        for x in range(-pad, -pad + K_size):
            Kernel[y + pad, x + pad] = np.exp(-(x**2 + y**2) / (2 * (sigma ** 2)))
    Kernel /= (2 * np.pi * sigma * sigma)
    # print(Kernel)
    # print(Kernel.sum())
    Kernel /= Kernel.sum()
    # print(Kernel)
    # print(Kernel.shape)


    for y in range(H):
        for x in range(W):
            for c in range(C):
                # print(x, y ,img[y:y+K_size, x:x+K_size, c])
                out[pad + y, pad + x, c] = np.sum(Kernel * img[y:y+K_size, x:x+K_size, c])

    out = out[pad:pad+H, pad:pad+W].astype(np.uint8)
    return out

img = cv2.imread("./image_01_10/imori_noise.jpg").astype(np.float32)
img_zero = zero_pad(img)
img_ans = gaussian(img_zero)
print(img)
print("ーーーーーーーーーーーーーーーーーーーーーーーーーーーーー")
print(img_ans)

cv2.imwrite("./image_01_10/answer9.jpg",img_ans)
cv2.imshow("result", img_ans)
cv2.waitKey(0)
cv2.destroyAllWindows()
