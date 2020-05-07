import cv2
import numpy as np


channel = 3
T = 8
K = 4
v_max = 255


def w(x, y, u, v):
    cu = 1.
    cv = 1.
    if u == 0:
        cu /= np.sqrt(2)
    if v == 0:
        cv /= np.sqrt(2)
    theta = np.pi / (2 * T)
    return (( 2 * cu * cv / T) * np.cos((2*x+1)*v*theta) * np.cos((2*y+1)*u*theta))


def DCT(img):
    H ,W,_ = img.shape

    out = np.zeros((H,W,channel), dtype=np.float32)

    for c in range(channel):
        for u in range(T):
            for v in range(T):
                for y in range(T):
                    for x in range(T):
                        out[u:u+H:T,v:v+W:T,c] += img[y:y+H:T,x:x+W:T,c] * w(x,y,u,v)

    return out


def IDCT(_G):
    H ,W,_ = _G.shape

    out = np.zeros((H,W,channel), dtype=np.float32)

    for c in range(channel):
        for y in range(T):
            for x in range(T):
                for u in range(K):
                    for v in range(K):
                        out[y:y+H:T,x:x+W:T,c] += _G[u:u+H:T,v:v+W:T,c] * w(x,y,u,v)

    out = np.clip(out, 0, 255)
    out = np.round(out).astype(np.uint8)
    return out


def PSNR(_img, _img_idft):
    H, W, C = _img.shape

    MSE = np.sum((_img - _img_idft)**2)/(H*W*C)
    print(MSE)
    result = 10*np.log10(v_max**2/MSE)

    return result

img = cv2.imread("./image_31_40/imori.jpg").astype(np.float32)

igm_dct = DCT(img)
img_ans = IDCT(igm_dct)
img_psnr = PSNR(img, img_ans)

print("=== bitrate ===")
print(8*(K**2)/(T**2))
print("===============")

print("=== PSNR ===")
print(img_psnr)
print("===============")

cv2.imwrite("./image_31_40/answer37.jpg", img_ans)
cv2.imshow("result", img_ans)
cv2.waitKey(0)
cv2.destroyAllWindows()
