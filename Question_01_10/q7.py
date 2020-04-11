import cv2
import numpy as np
R = 8


def pooling(_img):
    img = _img.copy()
    out = np.zeros_like(img, dtype = np.float32)

    low, column , color = img.shape

    low_r = low//R
    column_r = column//R

    cnt = 0
    for y in range(low_r):
        for x in range(column_r):
            cnt += 1
            print(img[y*R:(y+1)*R,x*R:(x+1)*R], y*R , x*R, cnt)
            # 青成分のpooling
            v_blue = np.mean(img[R*y:R*(y+1), R*x:R*(x+1), 0]).astype(np.int)
            out[y*R:(y+1)*R,x*R:(x+1)*R,0] = v_blue

            # 緑成分のpooling
            v_green = np.mean(img[R*y:R*(y+1), R*x:R*(x+1), 1]).astype(np.int)
            out[y*R:(y+1)*R,x*R:(x+1)*R,1] = v_green

            # 赤成分のpooling
            v_red = np.mean(img[R*y:R*(y+1), R*x:R*(x+1), 2]).astype(np.int)
            out[y*R:(y+1)*R,x*R:(x+1)*R,2] = v_red


            #これだとだめだった(行ごとの合計しかだしてくれない)
            # 例えば、[ 45.  44.  45.  59.  78.  64.  50.  39.] ってかんじで
            # v_red = sum(img[y*R:(y+1)*R,x*R:(x+1)*R,2])//R
            # out[y*R:(y+1)*R,x*R:(x+1)*R,2] = v_red

            # こうすればOK
            # v_red = sum(sum(img[y*R:(y+1)*R,x*R:(x+1)*R,2]))//(R**2)

            print(out[y*R:(y+1)*R,x*R:(x+1)*R], y*R , x*R, cnt)

    out = out.astype(np.uint8)
    return out

img = cv2.imread("./image_01_10/imori.jpg").astype(np.float32)

img_ans = pooling(img)

cv2.imwrite("./image_01_10/amswer7.jpg", img_ans)
cv2.imshow("result", img_ans)
cv2.waitKey(0)
cv2.destroyAllWindows()
