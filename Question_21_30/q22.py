import cv2
import numpy as np
import matplotlib.pyplot as plt

def option(_img, m0 = 128, s0 = 52):
    img = _img.copy()
    m = np.mean(_img)
    s = np.std(_img)

    img = s0/s*(_img - m)+m0
    img[img < 0] = 0
    img[img > 255] = 255
    img = img.astype(np.uint8)
    return img


img = cv2.imread("./image_21_30/imori_dark.jpg").astype(np.float)
# out = img.copy()
if len(img.shape) == 3:
    H,W,C = img.shape

else:
    img = np.expand_dims(img, axis = -1)
    H,W,C = img.shape

out = option(img)

plt.hist(out.ravel(), bins= 225, rwidth = 0.8, range = (0, 255))
plt.savefig("./image_21_30/answer22_2.jpg")
plt.show()

cv2.imwrite("./image_21_30/answer22_1.jpg", out)
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.destroyAllWindows()
