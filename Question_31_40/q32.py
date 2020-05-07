import cv2
import numpy as np
import sys
sys.path.append("..")
from Question_01_10.q2 import Gray

# def dft(_img):
#     H ,W ,C= _img.shape
#     K, L = W, H
#     img = _img.copy()
#     out = np.zeros((L,K,C)).astype(np.float)

#     y = np.arange(H).repeat(W).reshape(H,-1)
#     x = np.tile(np.arange(W),(H,1))

#     for c in range(C):
#         for l in range(L):
#             for k in range(K):
#                 out[l,k,c] = np.sum(img[:,:,c]*np.exp(-2j*np.pi*(x*k/W+y*l/H)))/np.sqrt( W* H)

#     print(out)
#     out = out.clip(0,255)
#     return out.astype(np.uint8)


# def idft(_img):
#     H ,W , C= _img.shape
#     img = _img.copy()
#     X, Y = W,H
#     out = np.zeros((Y,X,C)).astype(np.float)

#     l = np.arange(Y).repeat(X).reshape(Y,-1)
#     k = np.tile(np.arange(X), (Y,1))

#     for c in range(C):
#         for y in range(Y):
#             for x in range(X):
#                 out[y,x,c] = np.abs(np.sum(img[:,:,c]*np.exp(2j*np.pi*(k*x/W+l*y/H))))
#     out = out/(H*W)
#     out = out.clip(0,255)
#     return out.astype(np.uint8)



# DFT hyper-parameters
K, L = 128, 128
channel = 3


# DFT
def dft(img):
    H, W, _ = img.shape

    # Prepare DFT coefficient
    G = np.zeros((L, K, channel), dtype=np.complex)

    # prepare processed index corresponding to original image positions
    x = np.tile(np.arange(W), (H, 1))
    y = np.arange(H).repeat(W).reshape(H, -1)

    # dft
    for c in range(channel):
        for l in range(L):
            for k in range(K):
                G[l, k, c] = np.sum(img[..., c] * np.exp(-2j * np.pi * (x * k / K + y * l / L))) / np.sqrt(K * L)
                #for n in range(N):
                #    for m in range(M):
                #        v += gray[n, m] * np.exp(-2j * np.pi * (m * k / M + n * l / N))
                #G[l, k] = v / np.sqrt(M * N)

    return G

# IDFT
def idft(G):
    # prepare out image
    H, W, _ = G.shape
    out = np.zeros((H, W, channel), dtype=np.float32)

    # prepare processed index corresponding to original image positions
    x = np.tile(np.arange(W), (H, 1))
    y = np.arange(H).repeat(W).reshape(H, -1)

    # idft
    for c in range(channel):
        for l in range(H):
            for k in range(W):
                out[l, k, c] = np.abs(np.sum(G[..., c] * np.exp(2j * np.pi * (x * k / W + y * l / H)))) / np.sqrt(W * H)

    # clipping
    out = np.clip(out, 0, 255)
    out = out.astype(np.uint8)

    return out



img = cv2.imread("./image_31_40/imori.jpg").astype(np.float)
# img_gray = Gray(img)
img_gray = img
#下とpsは値が違う
img_ans_1 = np.fft.fft2(img_gray)
print(img_ans_1)
img_ans_2 = np.fft.ifft2(img_ans_1).astype(np.uint8)
# img_ans_2 = cv2.cvtColor(img_ans_2,cv2.COLOR_GRAY2RGB)
# B = img_ans_2[:,:,0].astype(np.float)/0.0722
# G = img_ans_2[:,:,1].astype(np.float)/0.7152
# R = img_ans_2[:,:,2].astype(np.float)/0.2126
# C_All = B+G+R

# H,W,C = img.shape
# C_All_2 = np.zeros((H,W,C))
# C_All_2[:,:,0] = C_All
# C_All_2[:,:,1] = C_All
# C_All_2[:,:,2] = C_All
# print(img_ans_2)


# bairitsu = C_All_2/img_ans_2.astype(np.float)
# print(bairitsu)
# print(bairitsu.shape)

# img_ans_2[:,:,0] = B/bairitsu[:,:,0]
# img_ans_2[:,:,1] = G/bairitsu[:,:,1]
# img_ans_2[:,:,2] = R/bairitsu[:,:,2]


# img_ans_1 = np.fft.fftshift(img_ans_1)
ps = (np.abs(img_ans_1) / np.abs(img_ans_1).max() * 255).astype(np.uint8)
print(ps)
print(img_ans_2)

# #上とpsは値がちがう
# img_ans_1 = dft(img)
# print(img_ans_1)
# img_ans_2 = idft(img_ans_1)
# ps = (np.abs(img_ans_1) / np.abs(img_ans_1).max() * 255).astype(np.uint8)
# print(ps)
# print(img_ans_2)


cv2.imwrite("./image_31_40/answer32_1.jpg", ps)
cv2.imshow("result", ps)
cv2.waitKey(0)
cv2.destroyWindow("result")

cv2.imwrite("./image_31_40/answer32_2.jpg", img_ans_2)
cv2.imshow("result", img_ans_2)
cv2.waitKey(0)
cv2.destroyAllWindows()
