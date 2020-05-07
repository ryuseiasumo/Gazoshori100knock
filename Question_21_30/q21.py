import cv2
import numpy as np
import matplotlib.pyplot as plt

def normalization(_img, a = 0, b = 255):
    img = _img.copy()
    c = np.min(img)
    d = np.max(img)
    print(c)
    # [c <= img & img < d]
    img = (b-a)/(d-c)*(img-c)+a
    img[img < a] = a
    img[img > b] = b

    img = img.astype(np.uint8)
    return img


img = cv2.imread("./image_21_30/imori_dark.jpg").astype(np.float)
# out = img.copy()
if len(img.shape) == 3:
    H,W,C = img.shape

else:
    img = np.expand_dims(img, axis = -1)
    H,W,C = img.shape

out = normalization(img)

plt.hist(out.ravel(), bins= 225, rwidth = 0.8, range = (0, 255))
plt.savefig("./image_21_30/answer21_2.jpg")
plt.show()

cv2.imwrite("./image_21_30/answer21_1.jpg", out)
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.destroyAllWindows()
