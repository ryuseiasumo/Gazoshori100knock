import cv2
import numpy as np
import sys
sys.path.append("..")
from Question_01_10.q2 import Gray

channel = 3

def band_pass_filter(_G, ratio = 0.1, ratio2 = 0.5):
    H, W, _ = _G.shape
    print(H,W)

    y = np.arange(H).repeat(W).reshape(H,-1)
    x = np.tile(np.arange(W), (H,1))

    # make filter
    _x = x - W//2
    _y = y - H//2
    r = np.sqrt(_x**2 + _y**2)
    mask = np.ones((H, W), dtype=np.float32)
    mask[(r < (W//2*ratio) ) | ((W // 2 * ratio2) < r)] = 0

    mask = np.repeat(mask, channel).reshape(H, W, channel)

    # filtering
    _G *= mask
    return _G


img = cv2.imread("./image_31_40/imori.jpg").astype(np.float32)

img_fft = np.fft.fftn(img)
# img_low = low_pass_filter(img_fft)
img_shift = np.fft.fftshift(img_fft)
# print(img_shift)
cv2.imshow("result", img_shift.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyWindow("result")

img_band = band_pass_filter(img_shift)
cv2.imshow("result", img_band.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyWindow("result")

# img_high = img_shift

f_ishift = np.fft.ifftshift(img_band)
img_ans = np.fft.ifftn(f_ishift)
img_ans = np.abs(img_ans)

print(img)
print(img_ans)

img_ans = img_ans.astype(np.uint8)

cv2.imwrite("./image_31_40/answer_35.jpg", img_ans)
cv2.imshow("result", img_ans)
cv2.waitKey(0)
cv2.destroyAllWindows()
