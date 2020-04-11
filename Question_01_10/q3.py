import cv2
import numpy as np

def Gray(img_color):
    B = img_color[:,:,0].copy()
    G = img_color[:,:,1].copy()
    R = img_color[:,:,2].copy()

    Y = 0.2126 * R + 0.7152 * G + 0.0722 * B
    Y = Y.astype(np.uint8)
    print(Y)

    return Y

def Nichika(img_gray):
    for i in range(len(img_gray)):
        for j in range(len(img_gray[i])):
            if img_gray[i][j] < 128:
                img_gray[i][j] = 0
            else:
                img_gray[i][j] = 255

    return img_gray

def Binarization(img_gray, th = 128):
    img_gray[img_gray < th] = 0     #ブロードキャスト
    img_gray[img_gray >= th] = 255
    return img_gray

img = cv2.imread("./image_01_10/imori.jpg").astype(np.float32)
print(img)
img_g = Gray(img)
img_ans = Binarization(img_g)
# img_ans = Nichika(img_g)

cv2.imwrite("./image_01_10/answers3.jpg", img_ans)
cv2.imshow("result", img_ans)
cv2.waitKey(0)
cv2.destroyAllWindows()
