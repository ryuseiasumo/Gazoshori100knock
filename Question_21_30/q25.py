import cv2
import numpy as np

def Nearest_Neighbor(_img, ax = 1.5, ay = 1.5):
    H, W ,C = _img.shape
    img = _img.copy().astype(np.float)

    aH = int(ay*H)
    aW = int(ax*W)

    y = np.arange(aH).repeat(aW).reshape(aH, -1)
    print(y)

    x = np.tile(np.arange(aW), (aH,1))
    print(x)

    y = np.round(y/ ay).astype(np.int)
    x = np.round(x/ ax).astype(np.int)

    print(y)
    print(x)

    out = img[y,x]

    print(img)
    print("ーーーーーーーーーーーーーーー")
    print(out)

    out = out.astype(np.uint8)
    return out


img = cv2.imread("./image_21_30/imori.jpg")
img_ans = Nearest_Neighbor(img)

cv2.imwrite("./image_21_30/answer25.jpg",img_ans)
cv2.imshow("result", img_ans)
cv2.waitKey(0)
cv2.destroyAllWindows()
