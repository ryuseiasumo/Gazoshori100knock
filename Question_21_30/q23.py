import cv2
import numpy as np
import matplotlib.pyplot as plt

def equalization(_img, z_max = 255):
    img = _img.copy()
    out = img.copy()
    if len(img.shape) == 3:
        H,W,C = img.shape
    else:
        img = np.expand_dims(img, axis = -1)
        H,W,C = img.shape

    S = H * W * C * 1.
    sum_h = 0

    for i in range(z_max+1):
        ind = np.where(img == i)
        sum_h += len(img[ind])
        out[ind] = z_max/S*sum_h

    out = out.astype(np.uint8)
    return out


img = cv2.imread("./image_21_30/imori_dark.jpg").astype(np.float)
# out = img.copy()
out = equalization(img)

plt.hist(out.ravel(), bins= 225, rwidth = 0.8, range = (0, 255))
plt.savefig("./image_21_30/answer22_2.jpg")
plt.show()

cv2.imwrite("./image_21_30/answer22_1.jpg", out)
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.destroyAllWindows()
