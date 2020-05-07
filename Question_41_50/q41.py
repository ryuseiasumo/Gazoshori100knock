import cv2
import numpy as np
import sys
sys.path.append("..")
from Question_01_10.q2 import Gray

img = cv2.imread("./image_41_50/imori.jpg").astype(np.float)
H, W, C = img.shape
img_gray = Gray(img)

# img_G = cv2.GaussianBlur(img_gray, ksize = (5,5), sigmaX = 1.4)

#0パディングをする場合
img_gray = np.pad(img_gray,(1,1), 'constant')
_img_G = cv2.GaussianBlur(img_gray, ksize = (5,5), sigmaX = 1.4)
img_G = np.clip(_img_G, 0, 255)
img_G = _img_G[1:H+1,1:W+1]
print(img_G)
print("ーーーーーーーーーーーーーー")

fx = cv2.Sobel(img_G, cv2.CV_32F, 1, 0, ksize=3)
fy = cv2.Sobel(img_G, cv2.CV_32F, 0, 1, ksize=3)
fx = np.clip(fx, 0, 255).astype(np.uint8)
fy = np.clip(fy, 0, 255).astype(np.uint8)

edge = np.sqrt(np.power(fx, 2) + np.power(fy, 2)).astype(np.uint8)
fx = np.maximum(fx, 1e-5) #下の式でfx = 0だとまずいから？

angle = np.arctan(fy / fx)

angle = angle / np.pi * 180
angle[angle < -22.5] = 180 + angle[angle < -22.5]

_angle = np.zeros_like(angle, dtype=np.uint8)
_angle[np.where(angle <= 22.5)] = 0
_angle[np.where((angle > 22.5) & (angle <= 67.5))] = 45
_angle[np.where((angle > 67.5) & (angle <= 112.5))] = 90
_angle[np.where((angle > 112.5) & (angle <= 157.5))] = 135

angle_ans = _angle.astype(np.uint8)


cv2.imwrite("./image_41_50/answer41_1.jpg",edge)
cv2.imshow("result", edge)
cv2.waitKey(0)
cv2.destroyWindow("result")

cv2.imwrite("./image_41_50/answer41_2.jpg",angle_ans)
cv2.imshow("result", angle_ans)
cv2.waitKey(0)
cv2.destroyAllWindows()
