import cv2
import numpy as np

def saisenka(_img):
    H, W, C = _img.shape
    out = np.zeros((H, W), dtype=np.int)
    out[_img[..., 0] > 0] = 1

    count = 1
    while count > 0:
        count = 0
        tmp = out.copy()
        #ラスタスキャン
        for y in range(H):
            for x in range(W):
                if tmp[y,x] == 0:
                    continue

                judge = 0
                #condition1
                if (tmp[y, min(x+1, W-1)] + tmp[max(y-1, 0), x] + tmp[y, max(x-1, 0)] + tmp[min(y+1, H-1), x]) < 4:
                    judge += 1

                #condition2
                s = 0
                if tmp[y,min(x+1, W-1)] - tmp[y,min(x+1, W-1)]*tmp[max(y-1,0),min(x+1, W-1)]*tmp[max(y-1,0),x]:
                    s += 1

                if tmp[max(y-1,0),x] - tmp[max(y-1,0),x]*tmp[max(y-1,0),max(x-1, 0)]*tmp[y,max(x-1, 0)]:
                    s += 1

                if tmp[y,max(x-1, 0)] - tmp[y,max(x-1, 0)]*tmp[min(y+1,H-1),max(x-1, 0)]*tmp[min(y+1,H-1),x]:
                    s += 1

                if tmp[min(y+1,H-1),x] - tmp[min(y+1,H-1),x]*tmp[min(y+1,H-1),min(x+1, W-1)]*tmp[y, min(x+1, W-1)]:
                    s += 1
                if s == 1:
                    judge += 1

                #condition3
                if np.sum(tmp[max(y-1, 0) : min(y+2, H), max(x-1, 0) : min(x+2, W)]) >= 4:
                    judge += 1  #condition2がなりたつってことは4近傍に1つだけ1がある。

                # if all conditions are satisfied
                if judge == 3:
                    out[y, x] = 0
                    count += 1

    out = out.astype(np.uint8) * 255

    return out




img = cv2.imread("./image_61_70/gazo.png")

img_ans = saisenka(img)

cv2.imwrite("./image_61_70/answer_63.jpg", img_ans)
cv2.imshow("result", img_ans)
cv2.waitKey(0)
cv2.destroyAllWindows()
