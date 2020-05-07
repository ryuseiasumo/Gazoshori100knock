import cv2
import numpy as np

def gamma(_img,c = 1,g = 2.2):
    img = _img.copy().astype(np.float)
    out = img.copy()

    print(img)
    print("ーーーーーーーーーーーー")
    img /= 255.
    out = (1/c * img) ** (1/g)
    out *= 255.


   # #正規化
    # a = 0
    # b = 255
    # d = np.min(out)
    # e = np.max(out)
    # out = (b-a)/(e-d)*(out-d) + a
    # img[out < a] = a
    # img[out > b] = b

    out = out.astype(np.uint8)
    print(out)
    print("ーーーーーーーーーーーー")
    return out


img = cv2.imread("./image_21_30/imori_gamma.jpg")
img_ans = gamma(img)

cv2.imwrite("./image_21_30/answer24.jpg", img_ans)
cv2.imshow("result", img_ans)
cv2.waitKey(0)
cv2.destroyAllWindows()

