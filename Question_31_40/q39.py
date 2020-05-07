import cv2
import numpy as np

def YCbCr_RGB(_img):
    B = _img[:,:,0]
    G = _img[:,:,1]
    R = _img[:,:,2]

    Y = 0.299 * R + 0.5870 * G + 0.114 * B
    Cb = -0.1687 * R - 0.3313 * G + 0.5 * B + 128
    Cr = 0.5 * R - 0.4187 * G - 0.0813 * B + 128

    Y = 0.7*Y

    R = Y + (Cr - 128) * 1.402
    G = Y - (Cb - 128) * 0.3441 - (Cr - 128) * 0.7139
    B = Y + (Cb - 128) * 1.7718

    out = np.zeros_like(_img)
    out[:,:,0] = B
    out[:,:,1] = G
    out[:,:,2] = R

    return out

img = cv2.imread("./image_31_40/imori.jpg").astype(np.float32)

img_ans = YCbCr_RGB(img).astype(np.uint8)

cv2.imwrite("./image_31_40/answer39.jpg", img_ans)
cv2.imshow("result", img_ans)
cv2.waitKey(0)
cv2.destroyAllWindows()
