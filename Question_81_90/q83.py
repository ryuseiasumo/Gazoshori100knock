import cv2
import numpy as np


img = cv2.imread("./image_81_90/thorino.jpg").astype(np.float)
img_gray = 0.2126 * img[..., 2] + 0.7152 * img[..., 1] + 0.0722 * img[..., 0]
H, W = img_gray.shape
k = 0.04
th = 0.1

sobelx = cv2.Sobel(img_gray,cv2.CV_64F,1,0,ksize=3).astype(np.float32)
sobely = cv2.Sobel(img_gray,cv2.CV_64F,0,1,ksize=3).astype(np.float32)

Ix2 = sobelx**2
Iy2 = sobely**2
IxIy = sobelx*sobely

dstIx2 = cv2.GaussianBlur(Ix2, ksize=(3, 3), sigmaX = 3)
dstIy2 = cv2.GaussianBlur(Iy2,ksize=(3, 3), sigmaX = 3)
dstIxIy = cv2.GaussianBlur(IxIy, ksize=(3, 3), sigmaX = 3)

detH = dstIx2*dstIy2  - dstIxIy**2
R = detH - k* ((Ix2 + Iy2) ** 2)

out = np.array((img_gray,img_gray,img_gray))
out = np.transpose(out, (1,2,0))
out[R >= np.max(R) * th] = [0, 0, 255]

img_ans = out.astype(np.uint8)


cv2.imwrite("./image_81_90/answer_83.jpg", img_ans)
cv2.imshow("result", img_ans)
cv2.waitKey(0)
cv2.destroyAllWindows()
