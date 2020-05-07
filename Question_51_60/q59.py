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

    out = out.astype(np.uint8)
    return out


def ConnectedComponent(_img):
    H, W = _img.shape
    label = np.zeros_like(_img)
    lookuptable = [0]
    l = 0
    for y in range(H):
        for x in range(W):
            if _img[y][x] == 255:
                r1 = 10000000000
                r2 = 10000000000
                r3 = 10000000000
                r4 = 10000000000

                if y != 0:
                    if _img[y-1][x] == 255:
                        r2 = label[y-1][x]

                    if x != 0:
                        if _img[y-1][x-1] == 255:
                            r1 = label[y-1][x-1]

                    if x != H-1:
                        if _img[y-1][x+1] == 255:
                            r3 = label[y-1][x+1]

                if x != 0:
                    if _img[y][x-1] == 255:
                        r4 = label[y][x-1]

                if r1 != 10000000000 or r2 != 10000000000 or r3 != 10000000000 or r4 != 10000000000:
                    r_select = min(r1, r2, r3, r4)
                    label[y][x] = r_select
                    if r1 != 10000000000 and r1 != r_select:
                        label[label == r1] = r_select
                        lookuptable[r1] = r_select

                    if r2 != 10000000000 and r2 != r_select:
                        label[label == r2] = r_select
                        lookuptable[r2] = r_select

                    if r3 != 10000000000 and r3 != r_select:
                        label[label == r3] = r_select
                        lookuptable[r3] = r_select

                    if r4 != 10000000000 and r4 != r_select:
                        label[label == r4] = r_select
                        lookuptable[r4] = r_select

                else:
                    l += 1
                    label[y][x] = l
                    lookuptable.append(l)

    print(label[label > 0])
    print(lookuptable)

    # lookuptableの整理。
    a = 0
    t = 0
    for i in range(len(lookuptable)):
        if lookuptable[i] == t:
            lookuptable[i] = a

        else:
            t = lookuptable[i]
            a += 1
            lookuptable[i] = a
    print("整理後")
    print(lookuptable)
    for i in range(len(lookuptable)):
        label[label == i] = lookuptable[i]

    print(label[label > 0])

    COLORS = [[0, 0, 255], [0, 255, 0], [255, 0, 0], [255, 255, 0]]
    out = np.zeros((H, W ,3))

    for c in range(len(COLORS)):
        out[label == (c+1)] = COLORS[c]

    out = out.astype(np.uint8)
    return out


img = cv2.imread("./image_51_60/seg.png").astype(np.float)
gray_img = 0.2126 * img[..., 2] + 0.7152 * img[..., 1] + 0.0722 * img[..., 0]
img2 = Otsu2(gray_img)

img_ans = ConnectedComponent(img2)

cv2.imwrite("./image_51_60/answer59.png", img_ans)
cv2.imshow("result", img_ans)
cv2.waitKey(0)
cv2.destroyAllWindows()
