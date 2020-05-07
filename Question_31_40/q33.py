import cv2
import numpy as np
import sys
sys.path.append("..")
from Question_01_10.q2 import Gray

channel = 3

def low_pass_filter(_G, ratio = 0.5):
    H, W, _ = _G.shape
    print(H,W)
    # G = np.zeros_like(_G)

    # #transfer
    # G[:H//2,:W//2] = _G[H//2:,W//2:]
    # G[:H//2,W//2:] = _G[H//2:,:W//2]
    # G[H//2:,:W//2] = _G[:H//2,:W//2]
    # G[H//2:,W//2:] = _G[:H//2,W//2]

    # cv2.imshow("result", G.astype(np.uint8))
    # cv2.waitKey(0)
    # cv2.destroyWindow("result")

    # get distance from center (H / 2, W / 2)
    y = np.arange(H).repeat(W).reshape(H,-1)
    x = np.tile(np.arange(W), (H,1))

    # make filter
    _x = x - W//2
    _y = y - H//2
    r = np.sqrt(_x**2 + _y**2)
    mask = np.ones((H, W), dtype=np.float32)
    mask = np.repeat(mask, channel).reshape(H, W, channel)

    mask[r > (W // 2 * ratio)] = 0

    # filtering
    _G *= mask

    # reverse original positions
    # _G[:H//2, :W//2] = G[H//2:, W//2:]
    # _G[:H//2, W//2:] = G[H//2:, :W//2]
    # _G[H//2:, :W//2] = G[:H//2, W//2:]
    # _G[H//2:, W//2:] = G[:H//2, :W//2]

    return _G


img = cv2.imread("./image_31_40/imori.jpg").astype(np.complex)
img_gray = img
img_fft = np.fft.fftn(img_gray)
# img_low = low_pass_filter(img_fft)
img_shift = np.fft.fftshift(img_fft)
cv2.imshow("result", img_shift.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyWindow("result")

img_low = low_pass_filter(img_shift)
cv2.imshow("result", img_low.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyWindow("result")

f_ishift = np.fft.ifftshift(img_low)


img_ans = np.fft.ifftn(f_ishift)
img_ans = np.abs(img_ans).astype(np.uint8)

cv2.imwrite("./image_31_40/answer_33.jpg", img_ans)
cv2.imshow("result", img_ans)
cv2.waitKey(0)
cv2.destroyAllWindows()
