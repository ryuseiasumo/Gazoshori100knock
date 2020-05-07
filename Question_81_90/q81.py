import cv2
import numpy as np


img = cv2.imread("./image_81_90/thorino.jpg").astype(np.float)
img_gray = 0.2126 * img[..., 2] + 0.7152 * img[..., 1] + 0.0722 * img[..., 0]
H, W = img_gray.shape

sobelx = cv2.Sobel(img_gray,cv2.CV_64F,1,0,ksize=3).astype(np.float32)
sobely = cv2.Sobel(img_gray,cv2.CV_64F,0,1,ksize=3).astype(np.float32)

Ix2 = sobelx**2
Iy2 = sobely**2
IxIy = sobelx*sobely


detH = Ix2*Iy2 - IxIy**2

out = np.array((img_gray,img_gray,img_gray))
out = np.transpose(out, (1,2,0))
print(out)


for y in range(H):
    for x in range(W):
        if detH[y][x] >= np.max(detH[max(0, y-1):min(y+2, H), max(0, x-1):min(x+2, W)]) and detH[y, x] > np.max(detH) * 0.1:
            out[y][x] = [0, 0, 225]

img_ans = out.astype(np.uint8)
cv2.imwrite("./image_81_90/answer_81.jpg", img_ans)
cv2.imshow("result", img_ans)
#cv2.imshow("result", sobelx.astype(np.uint8))  #縦方向検出
#cv2.imshow("result", sobely.astype(np.uint8))　#横方向検出
cv2.waitKey(0)
cv2.destroyAllWindows()
