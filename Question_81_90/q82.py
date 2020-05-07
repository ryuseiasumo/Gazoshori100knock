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

dstIx2 = cv2.GaussianBlur(Ix2, ksize=(3, 3), sigmaX = 3)
dstIy2 = cv2.GaussianBlur(Iy2,ksize=(3, 3), sigmaX = 3)
dstIxIy = cv2.GaussianBlur(IxIy, ksize=(3, 3), sigmaX = 3)

dstIx2 = (dstIx2 - dstIx2.min())/dstIx2.max() * 255.
dstIy2 = (dstIy2 - dstIy2.min())/dstIy2.max() * 255.
dstIxIy = (dstIxIy - dstIxIy.min())/dstIxIy.max() * 255.

cv2.imwrite("./image_81_90/answer_82_ix.jpg", dstIx2.astype(np.uint8))
cv2.imwrite("./image_81_90/answer_82_iy.jpg", dstIy2.astype(np.uint8))
cv2.imwrite("./image_81_90/answer_82_ixiy.jpg", dstIxIy.astype(np.uint8))

#cv2.imshow("result", dstIx2.astype(np.uint8))  #縦方向検出
cv2.imshow("result", dstIy2.astype(np.uint8)) #横方向検出
#cv2.imshow("result", dstIxIy.astype(np.uint8)) #横方向検出
cv2.waitKey(0)
cv2.destroyAllWindows()
