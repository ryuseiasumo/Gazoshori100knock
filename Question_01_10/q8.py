import cv2
import numpy as np

def pooling_max(_img, R = 8):
    img = _img.copy()
    out = np.zeros_like(_img,dtype = np.float32)

    low, column, color = img.shape

    low_r = low//R
    column_r = column//R

    for y in range(low_r):
        for x in range(column_r):
            for c in range(color):
                out[y*R : (y+1)*R , x*R : (x+1)*R ,c] = np.max(img[y*R : (y+1)*R , x*R : (x+1)*R ,c])

    out = out.astype(np.uint8)
    return out

img = cv2.imread("./image_01_10/imori.jpg").astype(np.float32)
img_ans = pooling_max(img)

cv2.imwrite("./image_01_10/answer8.jpg",img_ans)
cv2.imshow("result", img_ans)
cv2.waitKey(0)
cv2.destroyAllWindows()
