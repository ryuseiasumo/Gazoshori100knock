import cv2
import numpy as np

def musking(img):
    #BGR -> HSV
    # print(img)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    musk = np.zeros_like(img_hsv)
    musk_blue = np.zeros_like(img_hsv)

    musk_blue[(img_hsv[:,:,0] >= 90) & (img_hsv[:,:, 0] <= 130)] = 1
    musk = 1 - musk_blue

    out_hsv = musk * img_hsv

    out = cv2.cvtColor(out_hsv, cv2.COLOR_HSV2BGR)
    out = out.astype(np.uint8)
    return out

img = cv2.imread("./image_71_80/imori.jpg")
img_ans = musking(img)

cv2.imwrite("./image_71_80/answer_71.jpg",img_ans)
cv2.imshow("result", img_ans)
cv2.waitKey(0)
cv2.destroyAllWindows()
