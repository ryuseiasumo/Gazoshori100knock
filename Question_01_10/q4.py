import numpy as np
import cv2
import matplotlib.pyplot as plt

def Gray(img):
    gray = 0.2126 * img[:,:,2] + 0.7152 * img[:,:,1] + 0.0722 * img[:,:,0]
    gray = gray.astype(np.uint8)
    return gray

def binarization(gray, th):
    gray[gray < th] = 0
    gray[gray >= th] = 255
    return gray

def otsu_binarization(gray):
    class_kan_bunsan = 0
    for th in range(256):
        v0 = gray[gray < th]   #np.where()と同じ？ブロードキャスト
        w0 = len(v0)
        m0 = np.mean(v0)   #平均をだしている

        v1 = gray[gray >= th]
        w1 = len(v1)
        m1 = np.mean(v1)

        print(th, w0, m0, w1, m1)

        tmp = w0 * w1 * ((m0-m1)**2) /((w0 + w1) ** 2)
        if tmp > class_kan_bunsan :
            class_kan_bunsan = tmp
            th_max = th

    binary_ans = binarization(gray , th_max)
    print(th_max)
    return binary_ans


img = cv2.imread("./image_01_10/imori.jpg").astype(np.float32)
img_gray = Gray(img)
img_ans = otsu_binarization(img_gray)


cv2.imwrite("./image_01_10/answers4.jpg", img_ans)
cv2.imshow("result",img_ans)
cv2.waitKey(0)
cv2.destroyAllWindows()

# plt.hist(gray ,bins = 255, rwidth = 0.8, range = (0, 255)) #binsは階級数
# plt.xlabel('value')
# plt.ylabel('appearance')
# plt.show()
