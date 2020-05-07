import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append("..")
from Question_01_10.q2 import Gray
import math

img = cv2.imread('image_41_50/imori.jpg')
img = Gray(img).astype(np.uint8)

img_padded = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
img_gaussianed = cv2.GaussianBlur(img_padded, (5, 5), sigmaX=1.4)
img_gaussianed = np.clip(img_gaussianed, 0, 255).astype(np.uint8)

print(img_gaussianed)

sobel_x = cv2.Sobel(img_gaussianed.copy(), cv2.CV_32F, dx=1, dy=0, ksize=3)
sobel_y = cv2.Sobel(img_gaussianed.copy(), cv2.CV_32F, dx=0, dy=1, ksize=3)
sobel_x = np.clip(sobel_x, 0, 255).astype(np.uint8)
sobel_y = np.clip(sobel_y, 0, 255).astype(np.uint8)
#重み付き和



edge = np.sqrt(np.power(sobel_x, 2) + np.power(sobel_y, 2))
#edge = np.array(edge, dtype=np.float32)
sobel_x = np.maximum(sobel_x, 1e-5)
angle = np.arctan(sobel_y / sobel_x)
angle = angle * (180 / np.pi)
angle = np.where((angle <= 22.5) & (angle > -22.5), 0, angle)
angle = np.where((22.5 < angle) & (angle <= 67.5), 45, angle)
angle = np.where((67.5 < angle) & (angle <= 112.5), 90, angle)
angle = np.where((112.5 < angle) & (angle <= 157.5), 135, angle)

print(angle)

cv2.imwrite("image_41_50/test.jpg", edge.astype(np.uint8))
cv2.imwrite("image_41_50/test1.jpg", angle.astype(np.uint8))
cv2.imwrite("image_41_50/test2.jpg", angle)
