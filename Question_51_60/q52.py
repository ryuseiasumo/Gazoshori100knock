import cv2
import numpy as np

def Otsu2(_img):
    th = 0
    max_b = 0
    th_max = 0

    for th in range(255):
        H, W = _img.shape
        img = _img.copy()
        img[img >= th] = 255
        img[img < th] = 0

        img0 = img[img == 0]
        img1 = img[img == 255]

        w0 = len(img0)
        w1 = len(img1)

        m0 = np.mean(img0)
        m1 = np.mean(img1)

        b2 = w0*w1/((w0+w1)**2) * ((m0 - m1)**2)
        if b2 > max_b:
            max_b = b2
            th_max = th
            out = img

    print(th_max)

    out = out.astype(np.uint8)
    return out

def MorphologyDilate(_img2, count = 3):
    H,W = _img2.shape
    out = _img2.copy()

    for n in range(count):
        for y in range(H):
            for x in range(W):
                dy1, dy2, dx1 , dx2 = -1, 1, -1, 1
                if y == 0:
                    dy1 = 0
                if y == H-1:
                    dy2 = 0
                if x == 0:
                    dx1 = 0
                if x == W-1:
                    dx2 = 0

                if _img2[y+dy1][x] == 255 or _img2[y][x+dx1] == 255 or _img2[y][x+dx2] == 255 or _img2[y+dy2][x] == 255:
                    out[y][x] = 255

    out = out.astype(np.uint8)
    return out

def MorphologyErode(_img2, count = 3):
    H,W = _img2.shape
    out = _img2.copy()

    for n in range(count):
        for y in range(H):
            for x in range(W):
                dy1, dy2, dx1 , dx2 = -1, 1, -1, 1
                if y == 0:
                    dy1 = 0
                if y == H-1:
                    dy2 = 0
                if x == 0:
                    dx1 = 0
                if x == W-1:
                    dx2 = 0

                if _img2[y+dy1][x] == 0 or _img2[y][x+dx1] == 0 or _img2[y][x+dx2] == 0 or _img2[y+dy2][x] == 0:
                    out[y][x] = 0

    out = out.astype(np.uint8)
    return out


img = cv2.imread("./image_51_60/imori.jpg").astype(np.float)
gray_img = 0.2126 * img[..., 2] + 0.7152 * img[..., 1] + 0.0722 * img[..., 0]
img2 = Otsu2(gray_img)
cv2.imshow("result", img2)
cv2.waitKey(0)
cv2.destroyWindow("result")

img1 = MorphologyErode(img2)
img_O = MorphologyDilate(img1)

print(img_O)
img_ans = img2 - img_O
print(img_ans)

cv2.imwrite("./image_51_60/answer52.jpg",img_ans)
cv2.imshow("result", img_ans)
cv2.waitKey(0)
cv2.destroyAllWindows()
