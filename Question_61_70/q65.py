import cv2
import numpy as np

def ZhangSuen(_img):
    H, W, _ = _img.shape
    img = np.zeros((H,W)).astype(np.float)
    img[_img[..., 0] > 0] = 1
    out = 1 - img

    while True:
        s1 = []
        s2 = []
        #step1
        count = 0
        for y in range(1,H-1):
            for x in range(1,W-1):

                #condition1
                if out[y][x] > 0: #ここでの黒画素ってのは線ってこと？
                    continue

                #condition2
                f1 = 0
                if (out[y-1, x+1] - out[y-1, x]) == 1:
                    f1 += 1
                if (out[y, x+1] - out[y-1, x+1]) == 1:
                    f1 += 1
                if (out[y+1, x+1] - out[y, x+1]) == 1:
                    f1 += 1
                if (out[y+1, x] - out[y+1,x+1]) == 1:
                    f1 += 1
                if (out[y+1, x-1] - out[y+1, x]) == 1:
                    f1 += 1
                if (out[y, x-1] - out[y+1, x-1]) == 1:
                    f1 += 1
                if (out[y-1, x-1] - out[y, x-1]) == 1:
                    f1 += 1
                if (out[y-1, x] - out[y-1, x-1]) == 1:
                    f1 += 1

                if f1 != 1:
                    continue

                #condition3
                f2 = np.sum(out[y-1:y+2, x-1:x+2])
                if f2 < 2 or f2 > 6:
                    continue

                #conditon4
                f3 = out[y-1,x] + out[y+1,x] + out[y, x+1]
                if f3 < 1:
                    continue

                #conditon5
                f4 = out[y+1,x] + out[y, x+1] + out[y,x-1]
                if f4 < 1:
                    continue

                s1.append([y,x])

        for v in s1:
            out[v[0], v[1]] = 1

        #step2
        for y in range(1,H-1):
            for x in range(1,W-1):

                #condition1
                if out[y][x] > 0: #ここでの黒画素ってのは線ってこと？
                    continue

                #condition2
                f1 = 0
                if (out[y-1, x+1] - out[y-1, x]) == 1:
                    f1 += 1
                if (out[y, x+1] - out[y-1, x+1]) == 1:
                    f1 += 1
                if (out[y+1, x+1] - out[y, x+1]) == 1:
                    f1 += 1
                if (out[y+1, x] - out[y+1,x+1]) == 1:
                    f1 += 1
                if (out[y+1, x-1] - out[y+1, x]) == 1:
                    f1 += 1
                if (out[y, x-1] - out[y+1, x-1]) == 1:
                    f1 += 1
                if (out[y-1, x-1] - out[y, x-1]) == 1:
                    f1 += 1
                if (out[y-1, x] - out[y-1, x-1]) == 1:
                    f1 += 1

                if f1 != 1:
                    continue

                #condition3
                f2 = np.sum(out[y-1:y+2, x-1:x+2])
                if f2 < 2 or f2 > 6:
                    continue

                #conditon4
                f3 = out[y-1,x]  + out[y, x+1] + out[y,x-1]
                if f3 < 1:
                    continue

                #conditon5
                f4 = out[y-1,x] + out[y+1,x] + out[y,x-1]
                if f4 < 1:
                    continue

                s2.append([y,x])

        for v in s2:
            out[v[0], v[1]] = 1

        # if not any pixel is changed
        if len(s1) < 1 and len(s2) < 1:
            break

    out = 1-out
    out = out*255
    out = out.astype(np.uint8)
    return out

img = cv2.imread("./image_61_70/gazo.png").astype(np.float)

img_ans = ZhangSuen(img)

cv2.imwrite("./image_61_70/answer_65.png", img_ans)
cv2.imshow("result",img_ans)
cv2.waitKey(0)
cv2.destroyAllWindows()
